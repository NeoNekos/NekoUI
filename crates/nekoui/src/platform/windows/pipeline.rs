#![allow(unsafe_code)]

use core::mem::size_of;

use windows::Win32::Foundation::RECT;
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
use crate::render::{
    BOX_SHAPE_BORDER_COLOR_OFFSET, BOX_SHAPE_BORDER_WIDTHS_OFFSET, BOX_SHAPE_CORNER_RADII_OFFSET,
    BOX_SHAPE_FILL_COLOR_OFFSET, BOX_SHAPE_LOCAL_POSITION_OFFSET, BOX_SHAPE_POSITION_OFFSET,
    BOX_SHAPE_SIZE_OFFSET, PreparedFrameContext,
};
use crate::scene::BoxShape;
use crate::style::Color;

use super::box_shape::BoxShapeDraw;
use super::clip::PhysicalScissorRect;
use super::device::D3d11DeviceState;
use super::shaders::{box_shape_pixel_shader_bytes, box_shape_vertex_shader_bytes};

const INITIAL_VERTEX_CAPACITY: usize = 4096;

pub(super) struct D3d11BoxShapePipeline {
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    input_layout: ID3D11InputLayout,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    depth_stencil_state: ID3D11DepthStencilState,
    vertices: ID3D11Buffer,
    vertex_capacity: usize,
    vertex_scratch: Vec<BoxShapeVertex>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct BoxShapeVertex {
    position: [f32; 2],
    local_position: [f32; 2],
    size: [f32; 4],
    corner_radii: [f32; 4],
    border_widths: [f32; 4],
    fill_color: [f32; 4],
    border_color: [f32; 4],
}

impl D3d11BoxShapePipeline {
    pub(super) fn new(device: &D3d11DeviceState) -> NekoResult<Self> {
        let vertex_bytecode = box_shape_vertex_shader_bytes()?;
        let pixel_bytecode = box_shape_pixel_shader_bytes()?;
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
                        "D3D11 box shape vertex shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreatePixelShader(pixel_bytecode, None, Some(&mut pixel_shader))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 box shape pixel shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateInputLayout(&input_elements(), vertex_bytecode, Some(&mut input_layout))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 box shape input layout unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateBlendState(&blend_desc(), Some(&mut blend_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 box shape blend state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateRasterizerState(&rasterizer_desc(), Some(&mut rasterizer_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 box shape rasterizer state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateDepthStencilState(&depth_stencil_desc(), Some(&mut depth_stencil_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 box shape depth-stencil state unavailable: {error}"
                    ))
                })?;
        }

        Ok(Self {
            vertex_shader: vertex_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape vertex shader was not returned")
            })?,
            pixel_shader: pixel_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape pixel shader was not returned")
            })?,
            input_layout: input_layout.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape input layout was not returned")
            })?,
            blend_state: blend_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape blend state was not returned")
            })?,
            rasterizer_state: rasterizer_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape rasterizer state was not returned")
            })?,
            depth_stencil_state: depth_stencil_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 box shape depth-stencil state was not returned")
            })?,
            vertices: create_dynamic_vertex_buffer(device, INITIAL_VERTEX_CAPACITY)?,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            vertex_scratch: Vec::with_capacity(INITIAL_VERTEX_CAPACITY),
        })
    }

    pub(super) fn draw(
        &mut self,
        device: &D3d11DeviceState,
        target: &ID3D11RenderTargetView,
        context: PreparedFrameContext,
        rects: &[BoxShapeDraw],
        scissor: PhysicalScissorRect,
    ) -> NekoResult<usize> {
        build_vertices_into(rects, context, &mut self.vertex_scratch)?;
        let vertex_count = self.vertex_scratch.len();
        if vertex_count == 0 {
            return Ok(0);
        }
        let physical_size = context.physical_surface_size();
        self.ensure_vertex_capacity(device, vertex_count)?;
        write_buffer(device, &self.vertices, &self.vertex_scratch)?;
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: physical_size.width() as f32,
            Height: physical_size.height() as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let stride = size_of::<BoxShapeVertex>() as u32;
        let offset = 0_u32;
        let vertex_buffers = [Some(self.vertices.clone())];
        let rtvs = [Some(target.clone())];
        let scissor_rect = d3d11_rect(scissor);

        // SAFETY: All COM objects were created by this backend device, the vertex buffer contains
        // `vertex_count` tightly packed `BoxShapeVertex` values, the scissor was clamped for
        // the current framebuffer, and the RTV is current surface generation.
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
            device.context().RSSetScissorRects(Some(&[scissor_rect]));
            device
                .context()
                .OMSetDepthStencilState(&self.depth_stencil_state, 0);
            device.context().Draw(vertex_count as u32, 0);
        }

        Ok(vertex_count / 6)
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
                    "D3D11 box shape vertex buffer unavailable: {error}"
                ))
            })?;
    }
    buffer.ok_or_else(|| {
        NekoError::resource_failure("D3D11 box shape vertex buffer was not returned")
    })
}

fn write_buffer(
    device: &D3d11DeviceState,
    buffer: &ID3D11Buffer,
    vertices: &[BoxShapeVertex],
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
                NekoError::resource_failure(format!("D3D11 box shape vertex map failed: {error}"))
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
        .checked_mul(size_of::<BoxShapeVertex>())
        .ok_or_else(|| NekoError::resource_failure("D3D11 box shape vertex buffer is too large"))?;
    u32::try_from(byte_width)
        .map_err(|_| NekoError::resource_failure("D3D11 box shape vertex buffer exceeds u32"))
}

fn input_elements() -> [D3D11_INPUT_ELEMENT_DESC; 7] {
    [
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_POSITION_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 1,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_LOCAL_POSITION_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 2,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_SIZE_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 3,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_CORNER_RADII_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 4,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_BORDER_WIDTHS_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 5,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_FILL_COLOR_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 6,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: BOX_SHAPE_BORDER_COLOR_OFFSET,
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
        ScissorEnable: true.into(),
        MultisampleEnable: false.into(),
        AntialiasedLineEnable: false.into(),
    }
}

fn d3d11_rect(scissor: PhysicalScissorRect) -> RECT {
    RECT {
        left: scissor.left,
        top: scissor.top,
        right: scissor.right,
        bottom: scissor.bottom,
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

#[cfg(test)]
fn build_vertices(
    rects: &[BoxShapeDraw],
    context: PreparedFrameContext,
) -> NekoResult<Vec<BoxShapeVertex>> {
    let mut vertices = Vec::new();
    build_vertices_into(rects, context, &mut vertices)?;
    Ok(vertices)
}

fn build_vertices_into(
    rects: &[BoxShapeDraw],
    context: PreparedFrameContext,
    vertices: &mut Vec<BoxShapeVertex>,
) -> NekoResult<()> {
    vertices.clear();
    let required = box_shape_vertex_count(rects.len())?;
    reserve_box_shape_vertices(vertices, required)?;
    for draw in rects {
        push_box_shape(vertices, draw.rect, draw.shape, context);
    }
    Ok(())
}

fn reserve_box_shape_vertices(
    vertices: &mut Vec<BoxShapeVertex>,
    required: usize,
) -> NekoResult<()> {
    let additional = required.saturating_sub(vertices.len());
    vertices.try_reserve_exact(additional).map_err(|_| {
        NekoError::resource_failure("D3D11 box shape vertex scratch allocation failed")
    })
}

fn box_shape_vertex_count(rect_count: usize) -> NekoResult<usize> {
    let vertex_count = rect_count
        .checked_mul(6)
        .ok_or_else(|| NekoError::resource_failure("D3D11 box shape batch is too large"))?;
    let _ = vertex_buffer_byte_width(vertex_count)?;
    Ok(vertex_count)
}

fn push_box_shape(
    vertices: &mut Vec<BoxShapeVertex>,
    rect: LayoutRect,
    shape: BoxShape,
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
    let opacity = shape.opacity().as_f32();
    let fill_color = optional_color_to_rgba(shape.fill(), opacity);
    let border_color = optional_color_to_rgba(shape.border_color(), opacity);
    let size = [rect.width(), rect.height(), 0.0, 0.0];
    let corner_radii = corner_radii_payload(shape.corner_radius());
    let border_widths = border_widths_payload(shape.border_width());
    vertices.extend_from_slice(&[
        box_shape_vertex(
            [x0, y0],
            [0.0, 0.0],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
        box_shape_vertex(
            [x1, y0],
            [rect.width(), 0.0],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
        box_shape_vertex(
            [x1, y1],
            [rect.width(), rect.height()],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
        box_shape_vertex(
            [x0, y0],
            [0.0, 0.0],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
        box_shape_vertex(
            [x1, y1],
            [rect.width(), rect.height()],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
        box_shape_vertex(
            [x0, y1],
            [0.0, rect.height()],
            size,
            corner_radii,
            border_widths,
            fill_color,
            border_color,
        ),
    ]);
}

fn box_shape_vertex(
    position: [f32; 2],
    local_position: [f32; 2],
    size: [f32; 4],
    corner_radii: [f32; 4],
    border_widths: [f32; 4],
    fill_color: [f32; 4],
    border_color: [f32; 4],
) -> BoxShapeVertex {
    BoxShapeVertex {
        position,
        local_position,
        size,
        corner_radii,
        border_widths,
        fill_color,
        border_color,
    }
}

fn corner_radii_payload(radii: crate::style::CornerRadii<crate::style::Length>) -> [f32; 4] {
    [
        radii.top_left.as_px(),
        radii.top_right.as_px(),
        radii.bottom_right.as_px(),
        radii.bottom_left.as_px(),
    ]
}

fn border_widths_payload(widths: crate::style::Edges<crate::style::Length>) -> [f32; 4] {
    [
        widths.top.as_px(),
        widths.right.as_px(),
        widths.bottom.as_px(),
        widths.left.as_px(),
    ]
}
fn to_ndc_x(x: f32, logical_extent: f32) -> f32 {
    (x / logical_extent) * 2.0 - 1.0
}

fn to_ndc_y(y: f32, logical_extent: f32) -> f32 {
    1.0 - (y / logical_extent) * 2.0
}

fn optional_color_to_rgba(color: Option<Color>, opacity: f32) -> [f32; 4] {
    color.map_or([0.0, 0.0, 0.0, 0.0], |color| color_to_rgba(color, opacity))
}

fn color_to_rgba(color: Color, opacity: f32) -> [f32; 4] {
    let [red, green, blue, alpha] = color
        .to_current_backend_sdr_srgb_rgba()
        .expect("box shape pipeline only receives colors convertible to SDR sRGB");
    [red, green, blue, alpha * opacity]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::layout::{LayoutSize, Viewport};
    use crate::platform::PhysicalSize;
    use crate::render::BOX_SHAPE_VERTEX_STRIDE;
    use crate::style::{CornerRadii, Edges, Length, Opacity, opacity};
    use core::mem::offset_of;

    #[test]
    fn vertex_layout_matches_generated_shader_offsets() {
        assert_eq!(size_of::<BoxShapeVertex>() as u32, BOX_SHAPE_VERTEX_STRIDE);
        assert_eq!(
            offset_of!(BoxShapeVertex, position) as u32,
            BOX_SHAPE_POSITION_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, local_position) as u32,
            BOX_SHAPE_LOCAL_POSITION_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, size) as u32,
            BOX_SHAPE_SIZE_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, corner_radii) as u32,
            BOX_SHAPE_CORNER_RADII_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, border_widths) as u32,
            BOX_SHAPE_BORDER_WIDTHS_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, fill_color) as u32,
            BOX_SHAPE_FILL_COLOR_OFFSET
        );
        assert_eq!(
            offset_of!(BoxShapeVertex, border_color) as u32,
            BOX_SHAPE_BORDER_COLOR_OFFSET
        );
    }

    #[test]
    fn box_shape_blend_state_uses_straight_alpha_colors() {
        let desc = blend_desc();
        let target = desc.RenderTarget[0];

        assert_eq!(target.SrcBlend, D3D11_BLEND_SRC_ALPHA);
        assert_eq!(target.DestBlend, D3D11_BLEND_INV_SRC_ALPHA);
        assert_eq!(target.SrcBlendAlpha, D3D11_BLEND_ONE);
        assert_eq!(target.DestBlendAlpha, D3D11_BLEND_INV_SRC_ALPHA);
    }

    #[test]
    fn vertex_buffer_byte_width_rejects_overflow_before_unsafe_copy() {
        let max_vertices = (u32::MAX as usize) / size_of::<BoxShapeVertex>();
        let max_rects = max_vertices / 6;

        assert!(vertex_buffer_byte_width(max_vertices).is_ok());
        assert_eq!(
            vertex_buffer_byte_width(max_vertices + 1)
                .unwrap_err()
                .kind(),
            ErrorKind::ResourceFailure
        );
        assert_eq!(
            next_vertex_capacity(max_vertices + 1).unwrap_err().kind(),
            ErrorKind::ResourceFailure
        );
        assert!(box_shape_vertex_count(max_rects).is_ok());
        assert_eq!(
            box_shape_vertex_count(max_rects + 1).unwrap_err().kind(),
            ErrorKind::ResourceFailure
        );
    }

    #[test]
    fn vertex_scratch_capacity_failure_returns_resource_failure() {
        let mut vertices = Vec::new();

        let error = reserve_box_shape_vertices(&mut vertices, usize::MAX).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ResourceFailure);
    }

    #[test]
    fn vertex_scratch_partial_capacity_reaches_required_capacity() {
        let required = 12;
        let mut vertices = Vec::with_capacity(required - 1);

        reserve_box_shape_vertices(&mut vertices, required).unwrap();

        assert!(vertices.capacity() >= required);
    }

    #[test]
    fn logical_full_rect_maps_to_full_ndc_at_scale_one() {
        assert_full_logical_rect_maps_to_full_ndc(1.0, PhysicalSize::new(100, 50));
    }

    #[test]
    fn logical_full_rect_maps_to_full_ndc_at_scale_two() {
        assert_full_logical_rect_maps_to_full_ndc(2.0, PhysicalSize::new(200, 100));
    }

    #[test]
    fn vertex_colors_apply_box_shape_opacity_to_fill_and_border_alpha() {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let rects = [BoxShapeDraw {
            rect: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            shape: BoxShape::new(
                Some(Color::rgba(1, 2, 3, 128)),
                Some(Color::rgba(4, 5, 6, 64)),
                Edges::all(Length::ZERO),
                CornerRadii::all(Length::ZERO),
                opacity(0.5),
            ),
        }];

        let vertices = build_vertices(&rects, context).unwrap();

        assert_eq!(vertices.len(), 6);
        assert!((vertices[0].fill_color[3] - (128.0 / 255.0 * 0.5)).abs() < f32::EPSILON);
        assert!((vertices[0].border_color[3] - (64.0 / 255.0 * 0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn vertex_colors_pack_oklch_as_sdr_srgb() {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let rects = [BoxShapeDraw {
            rect: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            shape: BoxShape::new(
                Some(Color::oklch(0.5, 0.1, 120.0)),
                Some(Color::oklcha(0.6, 0.12, 40.0, 0.75)),
                Edges::all(Length::ZERO),
                CornerRadii::all(Length::ZERO),
                opacity(0.5),
            ),
        }];

        let vertices = build_vertices(&rects, context).unwrap();

        assert_eq!(vertices.len(), 6);
        assert_rgba_close(
            vertices[0].fill_color,
            [0.420_088_9, 0.328_834_68, 0.593_882_74, 0.5],
        );
        assert_rgba_close(
            vertices[0].border_color,
            [0.766_396_2, 0.265_599_55, 0.681_509_73, 0.375],
        );
    }

    #[test]
    fn missing_border_color_packs_transparent_border_with_requested_width() {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let rects = [BoxShapeDraw {
            rect: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            shape: BoxShape::new(
                Some(Color::rgb(1, 2, 3)),
                None,
                Edges::all(Length::Px(2.0)),
                CornerRadii::all(Length::ZERO),
                Opacity::OPAQUE,
            ),
        }];

        let vertices = build_vertices(&rects, context).unwrap();

        assert_eq!(vertices.len(), 6);
        assert_eq!(vertices[0].border_color, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(vertices[0].border_widths, [2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn non_uniform_border_widths_and_corner_radii_are_packed_per_side_and_corner() {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let rects = [BoxShapeDraw {
            rect: LayoutRect::new(0.0, 0.0, 40.0, 20.0),
            shape: BoxShape::new(
                Some(Color::rgb(1, 2, 3)),
                Some(Color::rgb(4, 5, 6)),
                Edges {
                    top: Length::Px(1.0),
                    right: Length::Px(2.0),
                    bottom: Length::Px(3.0),
                    left: Length::Px(4.0),
                },
                CornerRadii {
                    top_left: Length::Px(5.0),
                    top_right: Length::Px(6.0),
                    bottom_right: Length::Px(7.0),
                    bottom_left: Length::Px(8.0),
                },
                Opacity::OPAQUE,
            ),
        }];

        let vertices = build_vertices(&rects, context).unwrap();

        assert_eq!(vertices.len(), 6);
        assert_eq!(vertices[0].size, [40.0, 20.0, 0.0, 0.0]);
        assert_eq!(vertices[0].border_widths, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vertices[0].corner_radii, [5.0, 6.0, 7.0, 8.0]);
    }

    fn assert_full_logical_rect_maps_to_full_ndc(scale_factor: f32, physical_size: PhysicalSize) {
        let logical_size = LayoutSize::new(100.0, 50.0);
        let context = PreparedFrameContext::for_surface(
            Viewport::new(logical_size, scale_factor),
            physical_size,
            1,
        );
        let rects = [BoxShapeDraw {
            rect: LayoutRect::new(0.0, 0.0, logical_size.width(), logical_size.height()),
            shape: BoxShape::new(
                Some(Color::rgb(1, 2, 3)),
                None,
                Edges::all(Length::ZERO),
                CornerRadii::all(Length::ZERO),
                Opacity::OPAQUE,
            ),
        }];

        let vertices = build_vertices(&rects, context).unwrap();
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

    fn assert_rgba_close(actual: [f32; 4], expected: [f32; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 0.000_001,
                "{actual} != {expected}"
            );
        }
    }
}
