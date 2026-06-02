#![allow(unsafe_code)]

use core::mem::size_of;

use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_VERTEX_BUFFER, D3D11_BLEND_DESC,
    D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_OP_ADD, D3D11_BLEND_SRC_ALPHA,
    D3D11_BOX, D3D11_BUFFER_DESC, D3D11_COLOR_WRITE_ENABLE_ALL, D3D11_COMPARISON_ALWAYS,
    D3D11_COMPARISON_NEVER, D3D11_CPU_ACCESS_WRITE, D3D11_CULL_NONE, D3D11_DEPTH_STENCIL_DESC,
    D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_FILL_SOLID, D3D11_FILTER_MIN_MAG_MIP_LINEAR,
    D3D11_INPUT_ELEMENT_DESC, D3D11_INPUT_PER_VERTEX_DATA, D3D11_MAP_WRITE_DISCARD,
    D3D11_MAPPED_SUBRESOURCE, D3D11_RASTERIZER_DESC, D3D11_RENDER_TARGET_BLEND_DESC,
    D3D11_SAMPLER_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC_0,
    D3D11_SUBRESOURCE_DATA, D3D11_TEX2D_SRV, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC, D3D11_VIEWPORT, ID3D11BlendState, ID3D11Buffer,
    ID3D11DepthStencilState, ID3D11InputLayout, ID3D11PixelShader, ID3D11RasterizerState,
    ID3D11RenderTargetView, ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D,
    ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_SAMPLE_DESC,
};
use windows::core::s;

use crate::error::{NekoError, NekoResult};
use crate::render::{
    GLYPH_MONO_COLOR_OFFSET, GLYPH_MONO_GLYPH_ATLAS_D3D11_SRV_SLOT,
    GLYPH_MONO_GLYPH_SAMPLER_D3D11_SAMPLER_SLOT, GLYPH_MONO_POSITION_OFFSET, GLYPH_MONO_UV_OFFSET,
    PreparedFrame, PreparedFrameContext,
};
use crate::scene::SceneOrder;
use crate::style::Color;

use super::device::D3d11DeviceState;
use super::glyph::{
    GLYPH_ATLAS_HEIGHT, GLYPH_ATLAS_WIDTH, GlyphAtlas, GlyphDraw, GlyphUnsupportedReport,
    collect_glyph_draws,
};
use super::shaders::{glyph_mono_pixel_shader_bytes, glyph_mono_vertex_shader_bytes};

const INITIAL_VERTEX_CAPACITY: usize = 4096;

pub(super) struct D3d11GlyphMonoPipeline {
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    input_layout: ID3D11InputLayout,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    depth_stencil_state: ID3D11DepthStencilState,
    sampler: ID3D11SamplerState,
    atlas_texture: ID3D11Texture2D,
    atlas_view: ID3D11ShaderResourceView,
    vertices: ID3D11Buffer,
    vertex_capacity: usize,
    vertex_scratch: Vec<GlyphVertex>,
    glyph_draw_scratch: Vec<GlyphDraw>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GlyphVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

impl D3d11GlyphMonoPipeline {
    pub(super) fn new(device: &D3d11DeviceState) -> NekoResult<Self> {
        let vertex_bytecode = glyph_mono_vertex_shader_bytes()?;
        let pixel_bytecode = glyph_mono_pixel_shader_bytes()?;
        let mut vertex_shader = None;
        let mut pixel_shader = None;
        let mut input_layout = None;
        let mut blend_state = None;
        let mut rasterizer_state = None;
        let mut depth_stencil_state = None;
        let mut sampler = None;

        // SAFETY: The D3D11 device COM interface is owned by the Windows backend and remains
        // live for the pipeline lifetime. Shader bytecode comes from the checked DXBC artifact
        // accessors above, descriptor pointers reference fully initialized stack values, and all
        // output slots are valid `Option<T>` locations for COM objects returned by D3D11.
        unsafe {
            device
                .device()
                .CreateVertexShader(vertex_bytecode, None, Some(&mut vertex_shader))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph vertex shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreatePixelShader(pixel_bytecode, None, Some(&mut pixel_shader))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph pixel shader unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateInputLayout(&input_elements(), vertex_bytecode, Some(&mut input_layout))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph input layout unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateBlendState(&blend_desc(), Some(&mut blend_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph blend state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateRasterizerState(&rasterizer_desc(), Some(&mut rasterizer_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph rasterizer state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateDepthStencilState(&depth_stencil_desc(), Some(&mut depth_stencil_state))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph depth-stencil state unavailable: {error}"
                    ))
                })?;
            device
                .device()
                .CreateSamplerState(&sampler_desc(), Some(&mut sampler))
                .map_err(|error| {
                    NekoError::resource_failure(format!(
                        "D3D11 glyph sampler state unavailable: {error}"
                    ))
                })?;
        }
        let (atlas_texture, atlas_view) = create_atlas_resources(device)?;
        Ok(Self {
            vertex_shader: vertex_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph vertex shader was not returned")
            })?,
            pixel_shader: pixel_shader.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph pixel shader was not returned")
            })?,
            input_layout: input_layout.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph input layout was not returned")
            })?,
            blend_state: blend_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph blend state was not returned")
            })?,
            rasterizer_state: rasterizer_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph rasterizer state was not returned")
            })?,
            depth_stencil_state: depth_stencil_state.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph depth-stencil state was not returned")
            })?,
            sampler: sampler.ok_or_else(|| {
                NekoError::resource_failure("D3D11 glyph sampler state was not returned")
            })?,
            atlas_texture,
            atlas_view,
            vertices: create_dynamic_vertex_buffer(device, INITIAL_VERTEX_CAPACITY)?,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            vertex_scratch: Vec::with_capacity(INITIAL_VERTEX_CAPACITY),
            glyph_draw_scratch: Vec::new(),
        })
    }

    pub(super) fn collect_glyph_draws(
        &mut self,
        prepared: &PreparedFrame,
        atlas: &GlyphAtlas,
    ) -> NekoResult<GlyphUnsupportedReport> {
        collect_glyph_draws(prepared, atlas, &mut self.glyph_draw_scratch)
    }

    pub(super) fn glyph_draw_count(&self) -> usize {
        self.glyph_draw_scratch.len()
    }

    pub(super) fn glyph_draw_order(&self, index: usize) -> SceneOrder {
        self.glyph_draw_scratch[index].order
    }

    pub(super) fn draw_collected_range(
        &mut self,
        device: &D3d11DeviceState,
        target: &ID3D11RenderTargetView,
        context: PreparedFrameContext,
        range: std::ops::Range<usize>,
    ) -> NekoResult<usize> {
        let glyphs = &self.glyph_draw_scratch[range];
        build_vertices_into(glyphs, context, &mut self.vertex_scratch);
        self.draw_vertices(device, target, context)
    }

    fn draw_vertices(
        &mut self,
        device: &D3d11DeviceState,
        target: &ID3D11RenderTargetView,
        context: PreparedFrameContext,
    ) -> NekoResult<usize> {
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
        let stride = size_of::<GlyphVertex>() as u32;
        let offset = 0_u32;
        let vertex_buffers = [Some(self.vertices.clone())];
        let render_targets = [Some(target.clone())];
        let shader_resources = [Some(self.atlas_view.clone())];
        let samplers = [Some(self.sampler.clone())];

        // SAFETY: All pipeline COM objects and the target RTV were created from this backend
        // device, and the immediate context is owned and serialized by the Windows renderer for
        // this frame. The vertex buffer has been grown and written for `vertex_count` tightly
        // packed `GlyphVertex` values, the atlas SRV/sampler stay alive through the call, and the
        // RTV belongs to the current surface generation selected by the frame transaction.
        unsafe {
            device
                .context()
                .OMSetRenderTargets(Some(&render_targets), None);
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
            device.context().PSSetShaderResources(
                GLYPH_MONO_GLYPH_ATLAS_D3D11_SRV_SLOT,
                Some(&shader_resources),
            );
            device
                .context()
                .PSSetSamplers(GLYPH_MONO_GLYPH_SAMPLER_D3D11_SAMPLER_SLOT, Some(&samplers));
            device
                .context()
                .OMSetBlendState(&self.blend_state, None, u32::MAX);
            device.context().RSSetState(&self.rasterizer_state);
            device
                .context()
                .OMSetDepthStencilState(&self.depth_stencil_state, 0);
            device.context().Draw(vertex_count as u32, 0);
        }
        Ok(vertex_count / 6)
    }

    pub(super) fn upload_if_dirty(
        &mut self,
        device: &D3d11DeviceState,
        atlas: &mut GlyphAtlas,
    ) -> NekoResult<()> {
        if atlas.take_dirty() {
            upload_atlas(device, &self.atlas_texture, atlas)?;
        }
        Ok(())
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

fn create_atlas_resources(
    device: &D3d11DeviceState,
) -> NekoResult<(ID3D11Texture2D, ID3D11ShaderResourceView)> {
    let zeros = vec![0_u8; (GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT) as usize];
    let desc = D3D11_TEXTURE2D_DESC {
        Width: GLYPH_ATLAS_WIDTH,
        Height: GLYPH_ATLAS_HEIGHT,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let initial = D3D11_SUBRESOURCE_DATA {
        pSysMem: zeros.as_ptr().cast(),
        SysMemPitch: GLYPH_ATLAS_WIDTH,
        SysMemSlicePitch: GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT,
    };
    let mut texture = None;

    // SAFETY: The atlas texture descriptor uses fixed nonzero glyph-atlas dimensions and a
    // single-mip R8 texture. `zeros` contains exactly width * height bytes, and the pitch/slice
    // values describe that checked byte store for the duration of the CreateTexture2D call.
    unsafe {
        device
            .device()
            .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
            .map_err(|error| {
                NekoError::resource_failure(format!(
                    "D3D11 glyph atlas texture unavailable: {error}"
                ))
            })?;
    }
    let texture = texture
        .ok_or_else(|| NekoError::resource_failure("D3D11 glyph atlas texture was not returned"))?;
    let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8_UNORM,
        ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
            },
        },
    };
    let mut view = None;

    // SAFETY: `texture` is a live COM object returned by CreateTexture2D on the same backend
    // device, and `view_desc` matches its R8 texture format, 2D dimension, and one-mip layout.
    // The output slot is a valid `Option<ID3D11ShaderResourceView>` for D3D11 to initialize.
    unsafe {
        device
            .device()
            .CreateShaderResourceView(&texture, Some(&view_desc), Some(&mut view))
            .map_err(|error| {
                NekoError::resource_failure(format!("D3D11 glyph atlas SRV unavailable: {error}"))
            })?;
    }
    let view =
        view.ok_or_else(|| NekoError::resource_failure("D3D11 glyph atlas SRV was not returned"))?;
    Ok((texture, view))
}

fn upload_atlas(
    device: &D3d11DeviceState,
    texture: &ID3D11Texture2D,
    atlas: &GlyphAtlas,
) -> NekoResult<()> {
    if atlas.pixels().len() != (GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT) as usize {
        return Err(NekoError::resource_failure(
            "glyph atlas pixel store has unexpected size",
        ));
    }
    let box_all = D3D11_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: GLYPH_ATLAS_WIDTH,
        bottom: GLYPH_ATLAS_HEIGHT,
        back: 1,
    };

    // SAFETY: `texture` is the live glyph atlas texture owned by this pipeline, `box_all` covers
    // exactly the fixed atlas dimensions, and the length check above proves `atlas.pixels()` holds
    // width * height R8 bytes. The source pointer stays valid for the synchronous D3D11 upload, and
    // the immediate context is owned by the Windows renderer while this upload is recorded.
    unsafe {
        device.context().UpdateSubresource(
            texture,
            0,
            Some(&box_all),
            atlas.pixels().as_ptr().cast(),
            GLYPH_ATLAS_WIDTH,
            GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT,
        );
    }
    Ok(())
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

    // SAFETY: `desc.ByteWidth` is produced by the checked vertex-buffer byte-width helper for this
    // glyph vertex type, and the descriptor requests a dynamic D3D11 vertex buffer with CPU write
    // access. The device COM object is live and the output slot is valid for D3D11 initialization.
    unsafe {
        device
            .device()
            .CreateBuffer(&desc, None, Some(&mut buffer))
            .map_err(|error| {
                NekoError::resource_failure(format!(
                    "D3D11 glyph vertex buffer unavailable: {error}"
                ))
            })?;
    }
    buffer.ok_or_else(|| NekoError::resource_failure("D3D11 glyph vertex buffer was not returned"))
}

fn write_buffer(
    device: &D3d11DeviceState,
    buffer: &ID3D11Buffer,
    vertices: &[GlyphVertex],
) -> NekoResult<()> {
    let _ = vertex_buffer_byte_width(vertices.len())?;
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();

    // SAFETY: The buffer was created as a dynamic CPU-writable glyph vertex buffer, and callers
    // ensure its capacity with the same checked byte-width calculation used above. A successful Map
    // returns a destination with at least `size_of_val(vertices)` bytes; source and destination do
    // not overlap, the copy length is exactly the initialized vertex slice size, and Unmap is called
    // on the same renderer-owned immediate context before returning.
    unsafe {
        device
            .context()
            .Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))
            .map_err(|error| {
                NekoError::resource_failure(format!("D3D11 glyph vertex map failed: {error}"))
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
        .checked_mul(size_of::<GlyphVertex>())
        .ok_or_else(|| NekoError::resource_failure("D3D11 glyph vertex buffer is too large"))?;
    u32::try_from(byte_width)
        .map_err(|_| NekoError::resource_failure("D3D11 glyph vertex buffer exceeds u32"))
}

fn input_elements() -> [D3D11_INPUT_ELEMENT_DESC; 3] {
    [
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: GLYPH_MONO_POSITION_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 1,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: GLYPH_MONO_UV_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("LOC"),
            SemanticIndex: 2,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: GLYPH_MONO_COLOR_OFFSET,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

fn sampler_desc() -> D3D11_SAMPLER_DESC {
    D3D11_SAMPLER_DESC {
        Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D11_COMPARISON_NEVER,
        BorderColor: [0.0; 4],
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
    }
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

fn build_vertices_into(
    glyphs: &[GlyphDraw],
    context: PreparedFrameContext,
    vertices: &mut Vec<GlyphVertex>,
) {
    vertices.clear();
    let required = glyphs.len().saturating_mul(6);
    vertices.reserve(required);
    for glyph in glyphs {
        push_glyph(vertices, *glyph, context);
    }
}

#[cfg(test)]
fn build_vertices(glyphs: &[GlyphDraw], context: PreparedFrameContext) -> Vec<GlyphVertex> {
    let mut vertices = Vec::new();
    build_vertices_into(glyphs, context, &mut vertices);
    vertices
}

fn push_glyph(vertices: &mut Vec<GlyphVertex>, glyph: GlyphDraw, context: PreparedFrameContext) {
    let logical_size = context.logical_viewport_size();
    if glyph.rect.width() <= 0.0
        || glyph.rect.height() <= 0.0
        || logical_size.width() <= 0.0
        || logical_size.height() <= 0.0
        || !context.scale_factor().is_finite()
        || context.scale_factor() <= 0.0
        || context.physical_surface_size().is_zero()
    {
        return;
    }
    let x0 = to_ndc_x(glyph.rect.x(), logical_size.width());
    let y0 = to_ndc_y(glyph.rect.y(), logical_size.height());
    let x1 = to_ndc_x(glyph.rect.x() + glyph.rect.width(), logical_size.width());
    let y1 = to_ndc_y(glyph.rect.y() + glyph.rect.height(), logical_size.height());
    let color = color_to_rgba(glyph.color);
    let uv = glyph.uv;
    vertices.extend_from_slice(&[
        vertex(x0, y0, uv.left, uv.top, color),
        vertex(x1, y0, uv.right, uv.top, color),
        vertex(x1, y1, uv.right, uv.bottom, color),
        vertex(x0, y0, uv.left, uv.top, color),
        vertex(x1, y1, uv.right, uv.bottom, color),
        vertex(x0, y1, uv.left, uv.bottom, color),
    ]);
}

fn vertex(x: f32, y: f32, u: f32, v: f32, color: [f32; 4]) -> GlyphVertex {
    GlyphVertex {
        position: [x, y],
        uv: [u, v],
        color,
    }
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
        .expect("glyph pipeline only receives sRGB colors");
    [
        red as f32 / 255.0,
        green as f32 / 255.0,
        blue as f32 / 255.0,
        alpha as f32 / 255.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::super::glyph::GlyphUv;
    use super::*;
    use crate::layout::{LayoutRect, LayoutSize, Viewport};
    use crate::platform::PhysicalSize;
    use crate::render::GLYPH_MONO_VERTEX_STRIDE;
    use crate::scene::SceneOrder;
    use core::mem::offset_of;

    #[test]
    fn vertex_layout_matches_generated_shader_offsets() {
        assert_eq!(size_of::<GlyphVertex>() as u32, GLYPH_MONO_VERTEX_STRIDE);
        assert_eq!(
            offset_of!(GlyphVertex, position) as u32,
            GLYPH_MONO_POSITION_OFFSET
        );
        assert_eq!(offset_of!(GlyphVertex, uv) as u32, GLYPH_MONO_UV_OFFSET);
        assert_eq!(
            offset_of!(GlyphVertex, color) as u32,
            GLYPH_MONO_COLOR_OFFSET
        );
    }

    #[test]
    fn glyph_vertices_preserve_interior_uvs() {
        let context = PreparedFrameContext::for_surface(
            Viewport::new(LayoutSize::new(100.0, 50.0), 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let vertices = build_vertices(
            &[GlyphDraw {
                order: SceneOrder::new(1),
                rect: LayoutRect::new(0.0, 0.0, 10.0, 5.0),
                uv: GlyphUv {
                    left: 0.25,
                    top: 0.5,
                    right: 0.75,
                    bottom: 1.0,
                },
                color: Color::rgb(255, 128, 0),
            }],
            context,
        );

        assert_eq!(vertices.len(), 6);
        assert_eq!(vertices[0].uv, [0.25, 0.5]);
        assert_eq!(vertices[2].uv, [0.75, 1.0]);
        assert_eq!(vertices[0].position, [-1.0, 1.0]);
    }

    #[test]
    fn glyph_vertex_scratch_reuses_capacity_and_preserves_input_order() {
        let context = PreparedFrameContext::for_surface(
            Viewport::new(LayoutSize::new(100.0, 50.0), 1.0),
            PhysicalSize::new(100, 50),
            1,
        );
        let glyphs = [
            GlyphDraw {
                order: SceneOrder::new(1),
                rect: LayoutRect::new(0.0, 0.0, 10.0, 5.0),
                uv: GlyphUv {
                    left: 0.1,
                    top: 0.2,
                    right: 0.3,
                    bottom: 0.4,
                },
                color: Color::rgb(255, 0, 0),
            },
            GlyphDraw {
                order: SceneOrder::new(2),
                rect: LayoutRect::new(10.0, 0.0, 10.0, 5.0),
                uv: GlyphUv {
                    left: 0.5,
                    top: 0.6,
                    right: 0.7,
                    bottom: 0.8,
                },
                color: Color::rgb(0, 0, 255),
            },
        ];
        let mut vertices = Vec::with_capacity(12);

        build_vertices_into(&glyphs, context, &mut vertices);
        let capacity = vertices.capacity();

        assert_eq!(vertices.len(), 12);
        assert_eq!(vertices[0].uv, [0.1, 0.2]);
        assert_eq!(vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(vertices[6].uv, [0.5, 0.6]);
        assert_eq!(vertices[6].color, [0.0, 0.0, 1.0, 1.0]);

        build_vertices_into(&glyphs[..1], context, &mut vertices);

        assert_eq!(vertices.capacity(), capacity);
        assert_eq!(vertices.len(), 6);
        assert_eq!(vertices[0].uv, [0.1, 0.2]);
    }
}
