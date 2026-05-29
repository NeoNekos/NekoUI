#![allow(unsafe_code)]

use windows::Win32::Graphics::Direct3D11::{ID3D11RenderTargetView, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_PRESENT, DXGI_RGBA, DXGI_SCALING_NONE, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGISwapChain,
};
use windows::core::Interface;

use crate::error::{NekoError, NekoResult};
use crate::platform::{BackendSurfaceState, PhysicalSize, Renderability};

use super::device::D3d11DeviceState;
use super::window::NativeWindowHandle;

pub(super) const WINDOWS_BACKEND_CLEAR_COLOR: [f32; 4] = [0.02, 0.025, 0.03, 1.0];

pub(super) struct DxgiSurface {
    swap_chain: IDXGISwapChain,
    back_buffer: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,
    physical_size: PhysicalSize,
    generation: u64,
    state: BackendSurfaceState,
}

impl DxgiSurface {
    pub(super) fn create(
        device: &D3d11DeviceState,
        window: NativeWindowHandle,
        physical_size: PhysicalSize,
        generation: u64,
    ) -> NekoResult<Self> {
        if physical_size.is_zero() {
            return Err(NekoError::not_renderable(
                "cannot create DXGI surface for zero size",
            ));
        }
        let desc = swap_chain_desc(physical_size, DXGI_FORMAT_B8G8R8A8_UNORM);
        // SAFETY: The HWND is borrowed from the live winit window, the descriptor pointer is valid
        // for the duration of the call, and the D3D11 device is a live COM interface.
        let swap_chain_1 = unsafe {
            device.factory().CreateSwapChainForHwnd(
                device.device(),
                window.hwnd(),
                &desc,
                None,
                None,
            )
        }
        .map_err(|error| {
            NekoError::resource_failure(format!("DXGI swapchain unavailable: {error}"))
        })?;
        let background_color = swap_chain_background_color();
        // SAFETY: The swapchain was just created and the color pointer is valid for the call.
        unsafe { swap_chain_1.SetBackgroundColor(&background_color) }.map_err(|error| {
            NekoError::resource_failure(format!(
                "DXGI swapchain background color unavailable: {error}"
            ))
        })?;
        let swap_chain = swap_chain_1.cast::<IDXGISwapChain>().map_err(|error| {
            NekoError::resource_failure(format!("DXGI swapchain cast failed: {error}"))
        })?;

        let mut surface = Self {
            swap_chain,
            back_buffer: None,
            render_target_view: None,
            physical_size,
            generation,
            state: BackendSurfaceState::Ready,
        };
        surface.recreate_backbuffer_view(device)?;
        Ok(surface)
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn physical_size(&self) -> PhysicalSize {
        self.physical_size
    }

    pub(super) fn state(&self) -> BackendSurfaceState {
        self.state
    }

    pub(super) fn resize(
        &mut self,
        device: &D3d11DeviceState,
        physical_size: PhysicalSize,
        generation: u64,
        renderability: Renderability,
    ) -> NekoResult<()> {
        self.generation = generation;
        self.physical_size = physical_size;
        self.release_generation_resources(device);
        if physical_size.is_zero() || !renderability.is_renderable() {
            self.state = BackendSurfaceState::Suspended;
            return Ok(());
        }
        // SAFETY: Old RTV/backbuffer references were dropped and indirect bindings were cleared;
        // the swapchain is live and the requested size is nonzero.
        let resize_result = unsafe {
            self.swap_chain.ResizeBuffers(
                0,
                physical_size.width(),
                physical_size.height(),
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG(0),
            )
        };
        if let Err(error) = resize_result {
            self.state = BackendSurfaceState::Lost;
            return Err(NekoError::backend_lost(format!(
                "DXGI ResizeBuffers failed: {error}"
            )));
        }
        self.recreate_backbuffer_view(device)?;
        self.state = BackendSurfaceState::Ready;
        Ok(())
    }

    pub(super) fn destroy(&mut self, device: &D3d11DeviceState) {
        self.state = BackendSurfaceState::Destroyed;
        self.release_generation_resources(device);
    }

    pub(super) fn clear(&self, device: &D3d11DeviceState, color: [f32; 4]) -> NekoResult<()> {
        let rtv = self.render_target_view()?;
        let rtvs = [Some(rtv.clone())];
        // SAFETY: The RTV was created from the current backbuffer generation and the immediate
        // context is backend-owned on the event-loop thread.
        unsafe {
            device.context().OMSetRenderTargets(Some(&rtvs), None);
            device.context().ClearRenderTargetView(rtv, &color);
        }
        Ok(())
    }

    pub(super) fn render_target_view(&self) -> NekoResult<&ID3D11RenderTargetView> {
        self.render_target_view
            .as_ref()
            .ok_or_else(|| NekoError::not_renderable("DXGI surface has no render target view"))
    }

    pub(super) fn present(&self) -> NekoResult<()> {
        // SAFETY: Present is called only for a ready, current, nonzero surface after recording.
        let result = unsafe { self.swap_chain.Present(1, DXGI_PRESENT(0)) };
        result
            .ok()
            .map_err(|error| NekoError::backend_lost(format!("DXGI Present failed: {error}")))
    }

    fn recreate_backbuffer_view(&mut self, device: &D3d11DeviceState) -> NekoResult<()> {
        // SAFETY: The swapchain is live and buffer 0 is requested as the D3D11 backbuffer type.
        let back_buffer =
            unsafe { self.swap_chain.GetBuffer::<ID3D11Texture2D>(0) }.map_err(|error| {
                NekoError::resource_failure(format!("DXGI backbuffer unavailable: {error}"))
            })?;
        let mut rtv = None;
        // SAFETY: back_buffer is a live D3D11 texture from the swapchain; default RTV desc is valid.
        unsafe {
            device
                .device()
                .CreateRenderTargetView(&back_buffer, None, Some(&mut rtv))
        }
        .map_err(|error| NekoError::resource_failure(format!("D3D11 RTV unavailable: {error}")))?;
        self.render_target_view =
            Some(rtv.ok_or_else(|| NekoError::resource_failure("D3D11 RTV was not returned"))?);
        self.back_buffer = Some(back_buffer);
        Ok(())
    }

    fn release_generation_resources(&mut self, device: &D3d11DeviceState) {
        self.render_target_view = None;
        self.back_buffer = None;
        device.clear_state_and_flush();
    }
}

pub(super) fn swap_chain_desc(
    physical_size: PhysicalSize,
    format: DXGI_FORMAT,
) -> DXGI_SWAP_CHAIN_DESC1 {
    DXGI_SWAP_CHAIN_DESC1 {
        Width: physical_size.width(),
        Height: physical_size.height(),
        Format: format,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    }
}

pub(super) fn swap_chain_background_color() -> DXGI_RGBA {
    DXGI_RGBA {
        r: WINDOWS_BACKEND_CLEAR_COLOR[0],
        g: WINDOWS_BACKEND_CLEAR_COLOR[1],
        b: WINDOWS_BACKEND_CLEAR_COLOR[2],
        a: WINDOWS_BACKEND_CLEAR_COLOR[3],
    }
}
