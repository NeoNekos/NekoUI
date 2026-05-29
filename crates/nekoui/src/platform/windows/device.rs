#![allow(unsafe_code)]

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device,
    ID3D11DeviceContext,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter, IDXGIDevice, IDXGIFactory2,
};
use windows::core::Interface;

use crate::error::{NekoError, NekoResult};

pub(super) struct D3d11DeviceState {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    factory: IDXGIFactory2,
    adapter_summary: String,
}

impl D3d11DeviceState {
    pub(super) fn create() -> NekoResult<Self> {
        let mut device = None;
        let mut context = None;
        let feature_levels = [D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1];

        // SAFETY: The output pointers are valid for this call, the feature-level slice lives for
        // the call duration, and no software rasterizer module is supplied for hardware creation.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .map_err(|error| {
            NekoError::resource_failure(format!("D3D11 device unavailable: {error}"))
        })?;

        let device =
            device.ok_or_else(|| NekoError::resource_failure("D3D11 device was not returned"))?;
        let context =
            context.ok_or_else(|| NekoError::resource_failure("D3D11 context was not returned"))?;

        let dxgi_device = device.cast::<IDXGIDevice>().map_err(|error| {
            NekoError::resource_failure(format!("D3D11 device is not a DXGI device: {error}"))
        })?;
        // SAFETY: dxgi_device is a live COM interface returned by D3D11CreateDevice.
        let adapter = unsafe { dxgi_device.GetAdapter() }.map_err(|error| {
            NekoError::resource_failure(format!("DXGI adapter unavailable: {error}"))
        })?;
        // SAFETY: adapter is a live DXGI object and the requested parent interface is IDXGIFactory2.
        let factory = unsafe { adapter.GetParent::<IDXGIFactory2>() }
            .or_else(|_| {
                // SAFETY: CreateDXGIFactory1 initializes an IDXGIFactory2 pointer or returns HRESULT.
                unsafe { CreateDXGIFactory1::<IDXGIFactory2>() }
            })
            .map_err(|error| {
                NekoError::resource_failure(format!("DXGI factory unavailable: {error}"))
            })?;
        let adapter_summary = adapter_summary(&adapter);

        Ok(Self {
            device,
            context,
            factory,
            adapter_summary,
        })
    }

    pub(super) fn device(&self) -> &ID3D11Device {
        &self.device
    }

    pub(super) fn context(&self) -> &ID3D11DeviceContext {
        &self.context
    }

    pub(super) fn factory(&self) -> &IDXGIFactory2 {
        &self.factory
    }

    pub(super) fn adapter_summary(&self) -> &str {
        &self.adapter_summary
    }

    pub(super) fn clear_state_and_flush(&self) {
        // SAFETY: The immediate context belongs to this backend device and is only used on the
        // winit event-loop thread by this renderer; clearing/flushing releases indirect bindings.
        unsafe {
            self.context.ClearState();
            self.context.Flush();
        }
    }
}

fn adapter_summary(adapter: &IDXGIAdapter) -> String {
    // SAFETY: adapter is a live DXGI interface; GetDesc fills and returns a value on success.
    match unsafe { adapter.GetDesc() } {
        Ok(desc) => {
            let name_end = desc
                .Description
                .iter()
                .position(|code_unit| *code_unit == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..name_end]);
            format!(
                "{} vendor={:#06x} device={:#06x}",
                name.trim(),
                desc.VendorId,
                desc.DeviceId
            )
        }
        Err(_) => "unknown adapter".to_owned(),
    }
}
