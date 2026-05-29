#![allow(unsafe_code)]

use core::mem::size_of;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_VERTEX_BUFFER, D3D11_BLEND_DESC, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE,
    D3D11_BLEND_OP_ADD, D3D11_BLEND_SRC_ALPHA, D3D11_BUFFER_DESC, D3D11_COLOR_WRITE_ENABLE_ALL,
    D3D11_COMPARISON_ALWAYS, D3D11_CPU_ACCESS_WRITE, D3D11_CULL_NONE, D3D11_DEPTH_STENCIL_DESC,
    D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_FILL_SOLID, D3D11_INPUT_ELEMENT_DESC,
    D3D11_INPUT_PER_VERTEX_DATA, D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE,
    D3D11_RASTERIZER_DESC, D3D11_RENDER_TARGET_BLEND_DESC, D3D11_USAGE_DYNAMIC, D3D11_VIEWPORT,
    ID3D11BlendState, ID3D11Buffer, ID3D11DepthStencilState, ID3D11InputLayout, ID3D11PixelShader,
    ID3D11RasterizerState, ID3D11RenderTargetView, ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
};
use windows::core::s;

use crate::error::{NekoError, NekoResult};
use crate::layout::LayoutRect;
use crate::render::{PreparedFrameContext, SOLID_RECT_COLOR_OFFSET, SOLID_RECT_POSITION_OFFSET};
use crate::style::Color;

use super::device::D3d11DeviceState;
use super::shaders::{solid_rect_pixel_shader_bytes, solid_rect_vertex_shader_bytes};
use super::solid_rect::SolidRectDraw;

const INITIAL_VERTEX_CAPACITY: usize = 4096;

pub(super) struct D3d11SolidRectPipeline {
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    input_layout: ID3D11InputLayout,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    depth_stencil_state: ID3D11DepthStencilState,
    vertices: ID3D11Buffer,
    vertex_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SolidRectVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl D3d11SolidRectPipeline {
    pub(super) fn new(device: &D3d11DeviceState) -> NekoResult<Self> {
        let vertex_bytecode = solid_rect_vertex_shader_bytes()?;
        let pixel_bytecode = solid_rect_pixel_shader_bytes()?;
        let mut vertex_shader = None;
        let mut pixel_shader = None;
        let mut input_layout = None;
        let mut blend_state = None;
        let mut rasterizer_state = None;
        let mut depth_stencil_state = None;

        // SAFETY: Bytecode slices are immutable checked artifacts included in the binary, output
        // pointers are valid for the call, and the D3D11 device is owned by this backend.
        unsafe {
            device
                .device()
                .CreateVertexShader(vertex_bytecode, None, Some(&mut vertex_shader))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect vertex shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreatePixelShader(pixel_bytecode, None, Some(&mut pixel_shader))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect pixel shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateInputLayout(&input_elements(), vertex_bytecode, Some(&mut input_layout))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect input layout unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateBlendState(&blend_desc(), Some(&mut blend_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect blend state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateRasterizerState(&rasterizer_desc(), Some(&mut rasterizer_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect rasterizer state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateDepthStencilState(&depth_stencil_desc(), Some(&mut depth_stencil_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 solid rect depth-stencil state unavailable: {error}"
                    ))
                })?;
        }

        Ok(Self {
            vertex_shader: vertex_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect vertex shader was not returned")
            })?,
            pixel_shader: pixel_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect pixel shader was not returned")
            })?,
            input_layout: input_layout.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect input layout was not returned")
            })?,
            blend_state: blend_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect blend state was not returned")
            })?,
            rasterizer_state: rasterizer_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect rasterizer state was not returned")
            })?,
            depth_stencil_state: depth_stencil_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 solid rect depth-stencil state was not returned")
            })?,
            vertices: create_dynamic_vertex_buffer(device, INITIAL_VERTEX_CAPACITY)?,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
        })
    }

    pub(super) fn draw(
        &mut self,
        device: &D3d11DeviceState,
        target: &ID3D11RenderTargetView,
        context: PreparedFrameContext,
        rects: &[SolidRectDraw],
    ) -> NekoResult<usize> {
        let vertices = build_vertices(rects, context);
        if vertices.is_empty() {
            return Ok(0);
        }
        let physical_size = context.physical_surface_size();
        self.ensure_vertex_capacity(device, vertices.len())?;
        write_buffer(device, &self.vertices, &vertices)?;
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: physical_size.width() as f32,
            Height: physical_size.height() as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let stride = size_of::<SolidRectVertex>() as u32;
        let offset = 0_u32;
        let vertex_buffers = [Some(self.vertices.clone())];
        let rtvs = [Some(target.clone())];

        // SAFETY: All COM objects were created by this backend device, the vertex buffer contains
        // `vertices.len()` tightly packed `SolidRectVertex` values, and the RTV is current surface generation.
        unsafe {
            device.context().OMSetRenderTargets(Some(&rtvs), None);
            device.context().RSSetViewports(Some(&[viewport]));
            device.context().IASetInputLayout(&self.input_layout);
            device
                .context()
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            device.context().IASetVertexBuffers(
                0,
                1,
                Some(vertex_buffers.as_ptr()),
                Some(&stride),
                Some(&offset),
            );
            device.context().VSSetShader(&self.vertex_shader, None);
            device.context().PSSetShader(&self.pixel_shader, None);
            device
                .context()
                .OMSetBlendState(&self.blend_state, None, u32::MAX);
            device.context().RSSetState(&self.rasterizer_state);
            device
                .context()
                .OMSetDepthStencilState(&self.depth_stencil_state, 0);
            device.context().Draw(vertices.len() as u32, 0);
        }

        Ok(vertices.len() / 6)
    }

    fn ensure_vertex_capacity(
        &mut self,
        device: &D3d11DeviceState,
        required: usize,
    ) -> NekoResult<()> {
        if required <= self.vertex_capacity {
            return Ok(());
        }
        let next = next_vertex_capacity(required)?;
        self.vertices = create_dynamic_vertex_buffer(device, next)?;
        self.vertex_capacity = next;
        Ok(())
    }
}

fn create_dynamic_vertex_buffer(
    device: &D3d11DeviceState,
    vertex_capacity: usize,
) -> NekoResult<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: vertex_buffer_byte_width(vertex_capacity)?,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut buffer = None;
    // SAFETY: The buffer descriptor is initialized for a dynamic vertex buffer and the output slot is valid.
    unsafe {
        device
            .device()
            .CreateBuffer(&desc, None, Some(&mut buffer))
            .map_err(|error| {
                NekoError::resource_failure(format!(
                    "D3D11 solid rect vertex buffer unavailable: {error}"
                ))
            })?;
    }
    buffer.ok_or_else(|| {
        NekoError::resource_failure("D3D11 solid rect vertex buffer was not returned")
    })
}

fn write_buffer(
    device: &D3d11DeviceState,
    buffer: &ID3D11Buffer,
    vertices: &[SolidRectVertex],
) -> NekoResult<()> {
    let _ = vertex_buffer_byte_width(vertices.len())?;
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // SAFETY: The buffer was created as D3D11_USAGE_DYNAMIC with CPU write access;
    // callers grow it through the same checked byte-width calculation used above, so
    // the mapped destination has at least `size_of_val(vertices)` bytes. Unmap is
    // called before return.
    unsafe {
        device
            .context()
            .Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))
            .map_err(|error| {
                NekoError::resource_failure(format!("D3D11 solid rect vertex map failed: {error}"))
            })?;
        core::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            mapped.pData.cast::<u8>(),
            core::mem::size_of_val(vertices),
        );
        device.context().Unmap(buffer, 0);
    }
    Ok(())
}

fn next_vertex_capacity(required: usize) -> NekoResult<usize> {
    let capacity = required.checked_next_power_of_two().unwrap_or(required);
    vertex_buffer_byte_width(capacity)?;
    Ok(capacity)
}

fn vertex_buffer_byte_width(vertex_capacity: usize) -> NekoResult<u32> {
    let byte_width = vertex_capacity
        .checked_mul(size_of::<SolidRectVertex>())
        .ok_or_else(|| {
            NekoError::resource_failure("D3D11 solid rect vertex buffer is too large")
        })?;
    u32::try_from(byte_width)
        .map_err(|_| NekoError::resource_failure("D3D11 solid rect vertex buffer exceeds u32"))
}

fn input_elements() -> [D3D11_INPUT_ELEMENT_DESC; 2] {
    [
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: SOLID_RECT_POSITION_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: SOLID_RECT_COLOR_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

fn blend_desc() -> D3D11_BLEND_DESC {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
        BlendEnable: true.into(),
        SrcBlend: D3D11_BLEND_SRC_ALPHA,
        DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
        BlendOp: D3D11_BLEND_OP_ADD,
        SrcBlendAlpha: D3D11_BLEND_ONE,
        DestBlendAlpha: D3D11_BLEND_INV_SRC_ALPHA,
        BlendOpAlpha: D3D11_BLEND_OP_ADD,
        RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    desc
}

fn rasterizer_desc() -> D3D11_RASTERIZER_DESC {
    D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: false.into(),
        ScissorEnable: false.into(),
        MultisampleEnable: false.into(),
        AntialiasedLineEnable: false.into(),
    }
}

fn depth_stencil_desc() -> D3D11_DEPTH_STENCIL_DESC {
    D3D11_DEPTH_STENCIL_DESC {
        DepthEnable: false.into(),
        DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
        DepthFunc: D3D11_COMPARISON_ALWAYS,
        StencilEnable: false.into(),
        StencilReadMask: 0,
        StencilWriteMask: 0,
        FrontFace: Default::default(),
        BackFace: Default::default(),
    }
}

fn build_vertices(rects: &[SolidRectDraw], context: PreparedFrameContext) -> Vec<SolidRectVertex> {
    let mut vertices = Vec::with_capacity(rects.len().saturating_mul(6));
    for draw in rects {
        push_rect(&mut vertices, draw.rect, draw.color, context);
    }
    vertices
}

fn push_rect(
    vertices: &mut Vec<SolidRectVertex>,
    rect: LayoutRect,
    color: Color,
    context: PreparedFrameContext,
) {
    let logical_size = context.logical_viewport_size();
    if rect.width() <= 0.0
        || rect.height() <= 0.0
        || logical_size.width() <= 0.0
        || logical_size.height() <= 0.0
        || !context.scale_factor().is_finite()
        || context.scale_factor() <= 0.0
        || context.physical_surface_size().is_zero()
    {
        return;
    }
    let x0 = to_ndc_x(rect.x(), logical_size.width());
    let y0 = to_ndc_y(rect.y(), logical_size.height());
    let x1 = to_ndc_x(rect.x() + rect.width(), logical_size.width());
    let y1 = to_ndc_y(rect.y() + rect.height(), logical_size.height());
    let color = color_to_rgba(color);
    vertices.extend_from_slice(&[
        SolidRectVertex {
            position: [x0, y0],
            color,
        },
        SolidRectVertex {
            position: [x1, y0],
            color,
        },
        SolidRectVertex {
            position: [x1, y1],
            color,
        },
        SolidRectVertex {
            position: [x0, y0],
            color,
        },
        SolidRectVertex {
            position: [x1, y1],
            color,
        },
        SolidRectVertex {
            position: [x0, y1],
            color,
        },
    ]);
}

fn to_ndc_x(x: f32, logical_extent: f32) -> f32 {
    (x / logical_extent) * 2.0 - 1.0
}

fn to_ndc_y(y: f32, logical_extent: f32) -> f32 {
    1.0 - (y / logical_extent) * 2.0
}

fn color_to_rgba(color: Color) -> [f32; 4] {
    let (red, green, blue, alpha) = color
        .srgb_channels()
        .expect("solid rect pipeline only receives sRGB colors");
    [
        red as f32 / 255.0,
        green as f32 / 255.0,
        blue as f32 / 255.0,
        alpha as f32 / 255.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LayoutSize, Viewport};
    use crate::platform::PhysicalSize;
    use crate::render::SOLID_RECT_VERTEX_STRIDE;
    use core::mem::offset_of;

    #[test]
    fn vertex_layout_matches_manifest_offsets() {
        assert_eq!(
            size_of::<SolidRectVertex>() as u32,
            SOLID_RECT_VERTEX_STRIDE
        );
        assert_eq!(
            offset_of!(SolidRectVertex, position) as u32,
            SOLID_RECT_POSITION_OFFSET
        );
        assert_eq!(
            offset_of!(SolidRectVertex, color) as u32,
            SOLID_RECT_COLOR_OFFSET
        );
    }

    #[test]
    fn vertex_buffer_byte_width_rejects_overflow_before_unsafe_copy() {
        let max_vertices = (u32::MAX as usize) / size_of::<SolidRectVertex>();

        assert!(vertex_buffer_byte_width(max_vertices).is_ok());
        assert!(vertex_buffer_byte_width(max_vertices + 1).is_err());
        assert!(next_vertex_capacity(max_vertices + 1).is_err());
    }

    #[test]
    fn logical_full_rect_maps_to_full_ndc_at_scale_one() {
        assert_full_logical_rect_maps_to_full_ndc(1.0, PhysicalSize::new(100, 50));
    }

    #[test]
    fn logical_full_rect_maps_to_full_ndc_at_scale_two() {
        assert_full_logical_rect_maps_to_full_ndc(2.0, PhysicalSize::new(200, 100));
    }

    fn assert_full_logical_rect_maps_to_full_ndc(scale_factor: f32, physical_size: PhysicalSize) {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, scale_factor),
            physical_size,
            1,
        );
        let rects = [SolidRectDraw {
            rect: LayoutRect::new(0.0, 0.0, logical_size.width(), logical_size.height()),
            color: Color::rgb(1, 2, 3),
        }];

        let vertices = build_vertices(&rects, context);
        let positions = vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();

        assert_eq!(vertices.len(), 6);
        assert_eq!(positions[0], [-1.0, 1.0]);
        assert_eq!(positions[1], [1.0, 1.0]);
        assert_eq!(positions[2], [1.0, -1.0]);
        assert_eq!(positions[3], [-1.0, 1.0]);
        assert_eq!(positions[4], [1.0, -1.0]);
        assert_eq!(positions[5], [-1.0, -1.0]);
        assert_eq!(context.scale_factor(), scale_factor);
    }
}
