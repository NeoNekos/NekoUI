use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use cosmic_text::{CacheKey, Color as CosmicColor, SwashCache};
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferBindingType, ColorTargetState,
    ColorWrites, Device, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, SurfaceConfiguration,
    TextureFormat, TextureViewDescriptor, VertexBufferLayout, VertexState, VertexStepMode,
    vertex_attr_array,
};
use winit::window::Window as WinitWindow;

use crate::error::PlatformError;
use crate::platform::wgpu::atlas::{AtlasEntry, GlyphAtlas, GlyphAtlasKind};
use crate::platform::wgpu::context::WgpuContext;
use crate::scene::{
    ClipClass, CompiledScene, EffectClass, LogicalBatch, MaterialClass, Primitive, SceneNodeId,
};
use crate::style::Color;
use crate::text_system::TextSystem;
use crate::window::WindowSize;

const ATLAS_SIZE: u32 = 2048;

const QUAD_SHADER: &str = include_str!("shader/quad.wgsl");
const TEXT_SHADER: &str = include_str!("shader/text.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    viewport: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadInstance {
    rect: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextInstance {
    rect: [f32; 4],
    uv_rect: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorTextInstance {
    rect: [f32; 4],
    uv_rect: [f32; 4],
    alpha: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKey {
    Quad,
    MonoText,
    ColorText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureBindingKey {
    None,
    MonoGlyphAtlas(u32),
    ColorGlyphAtlas(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveBatch {
    pipeline_key: PipelineKey,
    texture_binding: TextureBindingKey,
    start: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct GpuBatch {
    pipeline_key: PipelineKey,
    texture_binding: TextureBindingKey,
    clip_class: ClipClass,
    clip_bounds: Option<crate::scene::LayoutBox>,
    effect_class: EffectClass,
    instance_range: Range<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchClipPolicy {
    None,
    Rect,
}

impl From<ClipClass> for BatchClipPolicy {
    fn from(value: ClipClass) -> Self {
        match value {
            ClipClass::None => Self::None,
            ClipClass::Rect => Self::Rect,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchEffectPolicy {
    None,
    Opacity,
}

impl From<EffectClass> for BatchEffectPolicy {
    fn from(value: EffectClass) -> Self {
        match value {
            EffectClass::None => Self::None,
            EffectClass::Opacity => Self::Opacity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectRenderPolicy {
    Direct,
    InlineOpacity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BatchSubmitState {
    pipeline_key: PipelineKey,
    texture_binding: TextureBindingKey,
    clip_policy: BatchClipPolicy,
    clip_bounds: Option<crate::scene::LayoutBox>,
    effect_policy: BatchEffectPolicy,
    effect_render_policy: EffectRenderPolicy,
}

impl From<&GpuBatch> for BatchSubmitState {
    fn from(batch: &GpuBatch) -> Self {
        let effect_policy: BatchEffectPolicy = batch.effect_class.into();
        Self {
            pipeline_key: batch.pipeline_key,
            texture_binding: batch.texture_binding,
            clip_policy: batch.clip_class.into(),
            clip_bounds: batch.clip_bounds,
            effect_policy,
            effect_render_policy: effect_render_policy(effect_policy),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Default)]
struct GpuBatchBuilder {
    batches: Vec<GpuBatch>,
}

impl GpuBatchBuilder {
    fn push(&mut self, batch: GpuBatch) {
        if batch.instance_range.is_empty() {
            return;
        }

        if let Some(previous) = self.batches.last_mut()
            && can_merge_gpu_batches(previous, &batch)
        {
            previous.instance_range.end = batch.instance_range.end;
            return;
        }

        self.batches.push(batch);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SceneWalkState {
    offset: [f32; 2],
    opacity: f32,
    clip: Option<crate::scene::LayoutBox>,
}

struct TextPrimitiveParams<'a> {
    bounds: &'a crate::scene::LayoutBox,
    layout: &'a crate::text_system::TextLayout,
    color: &'a Color,
    scene_state: SceneWalkState,
    batch: &'a LogicalBatch,
}

struct LogicalBatchCursor<'a> {
    batches: &'a [LogicalBatch],
    index: usize,
}

impl<'a> LogicalBatchCursor<'a> {
    fn new(batches: &'a [LogicalBatch]) -> Self {
        Self { batches, index: 0 }
    }

    fn batch_for_primitive(&mut self, primitive_index: u32) -> &'a LogicalBatch {
        while self.index + 1 < self.batches.len()
            && primitive_index >= self.batches[self.index].primitive_range.end
        {
            self.index += 1;
        }

        let batch = &self.batches[self.index];
        debug_assert!(batch.primitive_range.start <= primitive_index);
        debug_assert!(primitive_index < batch.primitive_range.end);
        batch
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RenderOutcome {
    Presented,
    Reconfigure,
    RecreateSurface,
    Unavailable,
}

pub struct WindowRenderState {
    surface: wgpu::Surface<'static>,
    config: SurfaceConfiguration,
    current_size: WindowSize,
    suspended: bool,
}

pub struct RenderSystem {
    context: WgpuContext,
    view_buffer: Buffer,
    view_bind_group_layout: BindGroupLayout,
    view_bind_group: BindGroup,
    quad_pipeline: RenderPipeline,
    mono_text_pipeline: RenderPipeline,
    color_text_pipeline: RenderPipeline,
    text_texture_bind_group_layout: BindGroupLayout,
    mono_atlas: GlyphAtlas,
    color_atlas: GlyphAtlas,
    swash_cache: SwashCache,
    quad_instances: Vec<QuadInstance>,
    mono_text_instances: Vec<TextInstance>,
    color_text_instances: Vec<ColorTextInstance>,
    gpu_batches: Vec<GpuBatch>,
    quad_instance_buffer: Buffer,
    mono_text_instance_buffer: Buffer,
    color_text_instance_buffer: Buffer,
    quad_instance_capacity: usize,
    mono_text_instance_capacity: usize,
    color_text_instance_capacity: usize,
}

impl RenderSystem {
    pub fn new(
        window: Arc<WinitWindow>,
        physical_size: WindowSize,
    ) -> Result<(Self, WindowRenderState), PlatformError> {
        let (context, surface) = WgpuContext::new(window)?;

        let view_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("nekoui_view_bind_group_layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let view_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nekoui_view_uniform"),
                contents: bytemuck::bytes_of(&ViewUniform {
                    viewport: [
                        physical_size.width.max(1) as f32,
                        physical_size.height.max(1) as f32,
                    ],
                    _pad: [0.0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let view_bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("nekoui_view_bind_group"),
            layout: &view_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });

        let text_texture_bind_group_layout = create_text_texture_bind_group_layout(&context.device);
        let mono_atlas = GlyphAtlas::new(
            &context.device,
            &text_texture_bind_group_layout,
            GlyphAtlasKind::Mono,
            ATLAS_SIZE.min(context.max_texture_size),
        )?;
        let color_atlas = GlyphAtlas::new(
            &context.device,
            &text_texture_bind_group_layout,
            GlyphAtlasKind::Color,
            ATLAS_SIZE.min(context.max_texture_size),
        )?;
        let quad_pipeline = create_quad_pipeline(
            &context.device,
            &view_bind_group_layout,
            TextureFormat::Bgra8UnormSrgb,
        );
        let mono_text_pipeline = create_mono_text_pipeline(
            &context.device,
            &view_bind_group_layout,
            &text_texture_bind_group_layout,
            TextureFormat::Bgra8UnormSrgb,
        );
        let color_text_pipeline = create_color_text_pipeline(
            &context.device,
            &view_bind_group_layout,
            &text_texture_bind_group_layout,
            TextureFormat::Bgra8UnormSrgb,
        );

        let quad_instance_capacity = 64;
        let mono_text_instance_capacity = 256;
        let color_text_instance_capacity = 64;
        let quad_instance_buffer = create_instance_buffer::<QuadInstance>(
            &context.device,
            "nekoui_quad_instances",
            quad_instance_capacity,
        );
        let mono_text_instance_buffer = create_instance_buffer::<TextInstance>(
            &context.device,
            "nekoui_mono_text_instances",
            mono_text_instance_capacity,
        );
        let color_text_instance_buffer = create_instance_buffer::<ColorTextInstance>(
            &context.device,
            "nekoui_color_text_instances",
            color_text_instance_capacity,
        );

        let mut render_system = Self {
            context,
            view_buffer,
            view_bind_group_layout,
            view_bind_group,
            quad_pipeline,
            mono_text_pipeline,
            color_text_pipeline,
            text_texture_bind_group_layout,
            mono_atlas,
            color_atlas,
            swash_cache: SwashCache::new(),
            quad_instances: Vec::new(),
            mono_text_instances: Vec::new(),
            color_text_instances: Vec::new(),
            gpu_batches: Vec::new(),
            quad_instance_buffer,
            mono_text_instance_buffer,
            color_text_instance_buffer,
            quad_instance_capacity,
            mono_text_instance_capacity,
            color_text_instance_capacity,
        };
        let render_state = render_system.create_window_state(surface, physical_size)?;
        render_system.rebuild_pipelines(render_state.config.format);
        Ok((render_system, render_state))
    }

    pub fn create_surface_for_window(
        &self,
        window: Arc<WinitWindow>,
    ) -> Result<wgpu::Surface<'static>, PlatformError> {
        self.context.create_surface_for_window(window)
    }

    pub fn create_window_state(
        &mut self,
        surface: wgpu::Surface<'static>,
        physical_size: WindowSize,
    ) -> Result<WindowRenderState, PlatformError> {
        let current_size = physical_size;
        let physical_size = self.surface_extent_for(physical_size);
        let config = surface
            .get_default_config(
                &self.context.adapter,
                physical_size.width,
                physical_size.height,
            )
            .ok_or_else(|| PlatformError::new("surface has no default configuration"))?;
        surface.configure(&self.context.device, &config);
        self.rebuild_pipelines(config.format);
        Ok(WindowRenderState {
            surface,
            config,
            current_size,
            suspended: false,
        })
    }

    pub fn resize(
        &mut self,
        state: &mut WindowRenderState,
        physical_size: WindowSize,
    ) -> Result<(), PlatformError> {
        state.current_size = physical_size;
        if physical_size.width == 0 || physical_size.height == 0 {
            state.suspended = true;
            return Ok(());
        }

        let physical_size = self.surface_extent_for(physical_size);
        if !state.suspended
            && state.config.width == physical_size.width
            && state.config.height == physical_size.height
        {
            return Ok(());
        }

        state.config.width = physical_size.width;
        state.config.height = physical_size.height;
        state.suspended = false;
        state.surface.configure(&self.context.device, &state.config);
        self.rebuild_pipelines(state.config.format);
        Ok(())
    }

    pub fn recreate_surface(
        &mut self,
        state: &mut WindowRenderState,
        window: Arc<WinitWindow>,
    ) -> Result<(), PlatformError> {
        state.surface = self.context.create_surface_for_window(window)?;
        state.suspended = false;
        self.resize(state, state.current_size)
    }

    pub fn render(
        &mut self,
        state: &mut WindowRenderState,
        scene: &CompiledScene,
        text_system: &mut TextSystem,
        window: &WinitWindow,
        scale_factor: f64,
    ) -> Result<RenderOutcome, PlatformError> {
        if state.current_size.width == 0 || state.current_size.height == 0 {
            return Ok(RenderOutcome::Unavailable);
        }

        let target_size = self.surface_extent_for(state.current_size);
        if state.suspended
            || state.config.width != target_size.width
            || state.config.height != target_size.height
        {
            self.resize(state, state.current_size)?;
        }

        self.context.queue.write_buffer(
            &self.view_buffer,
            0,
            bytemuck::bytes_of(&ViewUniform {
                viewport: [state.config.width as f32, state.config.height as f32],
                _pad: [0.0; 2],
            }),
        );

        self.quad_instances.clear();
        self.mono_text_instances.clear();
        self.color_text_instances.clear();
        self.gpu_batches.clear();
        self.mono_atlas.begin_frame();
        self.color_atlas.begin_frame();
        self.collect_instances(scene, text_system, scale_factor as f32);
        self.ensure_quad_capacity(self.quad_instances.len());
        self.ensure_mono_text_capacity(self.mono_text_instances.len());
        self.ensure_color_text_capacity(self.color_text_instances.len());

        if !self.quad_instances.is_empty() {
            self.context.queue.write_buffer(
                &self.quad_instance_buffer,
                0,
                bytemuck::cast_slice(&self.quad_instances),
            );
        }
        if !self.mono_text_instances.is_empty() {
            self.context.queue.write_buffer(
                &self.mono_text_instance_buffer,
                0,
                bytemuck::cast_slice(&self.mono_text_instances),
            );
        }
        if !self.color_text_instances.is_empty() {
            self.context.queue.write_buffer(
                &self.color_text_instance_buffer,
                0,
                bytemuck::cast_slice(&self.color_text_instances),
            );
        }

        let frame = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                drop(frame);
                self.resize(state, state.current_size)?;
                return Ok(RenderOutcome::Reconfigure);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(state, state.current_size)?;
                return Ok(RenderOutcome::Reconfigure);
            }
            wgpu::CurrentSurfaceTexture::Lost => return Ok(RenderOutcome::RecreateSurface),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(RenderOutcome::Unavailable);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(PlatformError::new(
                    "surface validation failed during get_current_texture",
                ));
            }
        };

        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nekoui_encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("nekoui_render_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(color_to_wgpu(
                            scene.clear_color.unwrap_or(Color::rgba(1.0, 1.0, 1.0, 1.0)),
                        )),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            let mut current_submit_state = None;
            for batch in &self.gpu_batches {
                let submit_state = BatchSubmitState::from(batch);
                if current_submit_state != Some(submit_state) {
                    if current_submit_state.map(|state| state.pipeline_key)
                        != Some(submit_state.pipeline_key)
                    {
                        match submit_state.pipeline_key {
                            PipelineKey::Quad => {
                                pass.set_pipeline(&self.quad_pipeline);
                                pass.set_bind_group(0, &self.view_bind_group, &[]);
                                pass.set_vertex_buffer(0, self.quad_instance_buffer.slice(..));
                            }
                            PipelineKey::MonoText => {
                                pass.set_pipeline(&self.mono_text_pipeline);
                                pass.set_bind_group(0, &self.view_bind_group, &[]);
                                pass.set_vertex_buffer(0, self.mono_text_instance_buffer.slice(..));
                            }
                            PipelineKey::ColorText => {
                                pass.set_pipeline(&self.color_text_pipeline);
                                pass.set_bind_group(0, &self.view_bind_group, &[]);
                                pass.set_vertex_buffer(
                                    0,
                                    self.color_text_instance_buffer.slice(..),
                                );
                            }
                        }
                    }

                    if current_submit_state.map(|state| state.texture_binding)
                        != Some(submit_state.texture_binding)
                    {
                        match submit_state.texture_binding {
                            TextureBindingKey::None => {}
                            TextureBindingKey::MonoGlyphAtlas(page_id) => {
                                let Some(bind_group) = self.mono_atlas.bind_group(page_id) else {
                                    continue;
                                };
                                pass.set_bind_group(1, bind_group, &[]);
                            }
                            TextureBindingKey::ColorGlyphAtlas(page_id) => {
                                let Some(bind_group) = self.color_atlas.bind_group(page_id) else {
                                    continue;
                                };
                                pass.set_bind_group(1, bind_group, &[]);
                            }
                        }
                    }

                    match submit_state.clip_policy {
                        BatchClipPolicy::None => {
                            pass.set_scissor_rect(0, 0, state.config.width, state.config.height);
                        }
                        BatchClipPolicy::Rect => {
                            let Some(scissor) = clip_bounds_to_scissor_rect(
                                submit_state.clip_bounds,
                                scale_factor as f32,
                                state.config.width,
                                state.config.height,
                            ) else {
                                continue;
                            };
                            pass.set_scissor_rect(
                                scissor.x,
                                scissor.y,
                                scissor.width,
                                scissor.height,
                            );
                        }
                    }

                    current_submit_state = Some(submit_state);
                }
                match submit_state.effect_render_policy {
                    EffectRenderPolicy::Direct => {
                        draw_gpu_batch(&mut pass, batch.instance_range.clone());
                    }
                    EffectRenderPolicy::InlineOpacity => {
                        draw_gpu_batch_inline_opacity(&mut pass, batch.instance_range.clone());
                    }
                }
            }
        }

        window.pre_present_notify();
        self.context.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(RenderOutcome::Presented)
    }

    fn collect_instances(
        &mut self,
        scene: &CompiledScene,
        text_system: &mut TextSystem,
        scale_factor: f32,
    ) {
        let scale_factor = scale_factor.max(f32::MIN_POSITIVE);
        if scene.scene_nodes.is_empty()
            || scene.primitives.is_empty()
            || scene.logical_batches.is_empty()
        {
            return;
        }

        let mut batch_cursor = LogicalBatchCursor::new(&scene.logical_batches);
        self.collect_node_instances(
            scene,
            text_system,
            scale_factor,
            SceneNodeId(0),
            SceneWalkState {
                offset: [0.0, 0.0],
                opacity: 1.0,
                clip: None,
            },
            &mut batch_cursor,
        );
    }

    fn ensure_glyph_entry(
        &mut self,
        text_system: &mut TextSystem,
        cache_key: CacheKey,
    ) -> Option<(GlyphAtlasKind, AtlasEntry)> {
        if let Some(entry) = self.mono_atlas.get(&cache_key) {
            return Some((GlyphAtlasKind::Mono, entry));
        }
        if let Some(entry) = self.color_atlas.get(&cache_key) {
            return Some((GlyphAtlasKind::Color, entry));
        }

        let image = self
            .swash_cache
            .get_image(text_system.font_system_mut(), cache_key)
            .as_ref()?
            .clone();

        match image.content {
            cosmic_text::SwashContent::Color => self
                .color_atlas
                .upload_color(&self.context.device, &self.context.queue, cache_key, &image)
                .map(|entry| (GlyphAtlasKind::Color, entry)),
            cosmic_text::SwashContent::Mask | cosmic_text::SwashContent::SubpixelMask => self
                .mono_atlas
                .upload_mask(&self.context.device, &self.context.queue, cache_key, &image)
                .map(|entry| (GlyphAtlasKind::Mono, entry)),
        }
    }

    fn surface_extent_for(&self, physical_size: WindowSize) -> WindowSize {
        let max = self.context.max_texture_size;
        WindowSize::new(
            physical_size.width.max(1).min(max),
            physical_size.height.max(1).min(max),
        )
    }

    fn rebuild_pipelines(&mut self, surface_format: TextureFormat) {
        self.quad_pipeline = create_quad_pipeline(
            &self.context.device,
            &self.view_bind_group_layout,
            surface_format,
        );
        self.mono_text_pipeline = create_mono_text_pipeline(
            &self.context.device,
            &self.view_bind_group_layout,
            &self.text_texture_bind_group_layout,
            surface_format,
        );
        self.color_text_pipeline = create_color_text_pipeline(
            &self.context.device,
            &self.view_bind_group_layout,
            &self.text_texture_bind_group_layout,
            surface_format,
        );
    }

    fn ensure_quad_capacity(&mut self, count: usize) {
        if count <= self.quad_instance_capacity {
            return;
        }
        while self.quad_instance_capacity < count {
            self.quad_instance_capacity *= 2;
        }
        self.quad_instance_buffer = create_instance_buffer::<QuadInstance>(
            &self.context.device,
            "nekoui_quad_instances",
            self.quad_instance_capacity,
        );
    }

    fn ensure_mono_text_capacity(&mut self, count: usize) {
        if count <= self.mono_text_instance_capacity {
            return;
        }
        while self.mono_text_instance_capacity < count {
            self.mono_text_instance_capacity *= 2;
        }
        self.mono_text_instance_buffer = create_instance_buffer::<TextInstance>(
            &self.context.device,
            "nekoui_mono_text_instances",
            self.mono_text_instance_capacity,
        );
    }

    fn ensure_color_text_capacity(&mut self, count: usize) {
        if count <= self.color_text_instance_capacity {
            return;
        }
        while self.color_text_instance_capacity < count {
            self.color_text_instance_capacity *= 2;
        }
        self.color_text_instance_buffer = create_instance_buffer::<ColorTextInstance>(
            &self.context.device,
            "nekoui_color_text_instances",
            self.color_text_instance_capacity,
        );
    }

    fn start_or_switch_batch(
        &mut self,
        active_batch: &mut Option<ActiveBatch>,
        pipeline_key: PipelineKey,
        texture_binding: TextureBindingKey,
        clip_class: ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: EffectClass,
    ) {
        let next_batch = ActiveBatch {
            pipeline_key,
            texture_binding,
            start: self.instance_count_for(pipeline_key),
        };

        if matches!(
            active_batch,
            Some(active)
                if active.pipeline_key == next_batch.pipeline_key
                    && active.texture_binding == next_batch.texture_binding
        ) {
            return;
        }

        self.finish_active_batch(active_batch, clip_class, clip_bounds, effect_class);
        *active_batch = Some(next_batch);
    }

    fn finish_active_batch(
        &mut self,
        active_batch: &mut Option<ActiveBatch>,
        clip_class: ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: EffectClass,
    ) {
        let Some(active_batch) = active_batch.take() else {
            return;
        };

        self.push_gpu_batch(
            active_batch.pipeline_key,
            active_batch.texture_binding,
            clip_class,
            clip_bounds,
            effect_class,
            active_batch.start..self.instance_count_for(active_batch.pipeline_key),
        );
    }

    fn push_gpu_batch(
        &mut self,
        pipeline_key: PipelineKey,
        texture_binding: TextureBindingKey,
        clip_class: ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: EffectClass,
        instance_range: Range<u32>,
    ) {
        push_gpu_batch(
            &mut self.gpu_batches,
            GpuBatch {
                pipeline_key,
                texture_binding,
                clip_class,
                clip_bounds,
                effect_class,
                instance_range,
            },
        );
    }

    fn instance_count_for(&self, pipeline_key: PipelineKey) -> u32 {
        match pipeline_key {
            PipelineKey::Quad => self.quad_instances.len() as u32,
            PipelineKey::MonoText => self.mono_text_instances.len() as u32,
            PipelineKey::ColorText => self.color_text_instances.len() as u32,
        }
    }

    fn collect_node_instances(
        &mut self,
        scene: &CompiledScene,
        text_system: &mut TextSystem,
        scale_factor: f32,
        node_id: SceneNodeId,
        parent_state: SceneWalkState,
        batch_cursor: &mut LogicalBatchCursor<'_>,
    ) {
        let node = &scene.scene_nodes[node_id.0 as usize];
        let current_offset = [
            parent_state.offset[0] + node.transform.tx,
            parent_state.offset[1] + node.transform.ty,
        ];
        let current_opacity = parent_state.opacity * node.opacity;
        let local_clip = node.clip.bounds.map(|bounds| crate::scene::LayoutBox {
            x: bounds.x + current_offset[0],
            y: bounds.y + current_offset[1],
            width: bounds.width,
            height: bounds.height,
        });
        let current_state = SceneWalkState {
            offset: current_offset,
            opacity: current_opacity,
            clip: intersect_clip(parent_state.clip, local_clip),
        };

        for primitive_index in node.primitive_range.as_range() {
            let batch = batch_cursor.batch_for_primitive(primitive_index as u32);
            match &scene.primitives[primitive_index] {
                Primitive::Quad { bounds, color } => {
                    debug_assert_eq!(batch.material_class, MaterialClass::Quad);
                    let start = self.quad_instances.len() as u32;
                    let rect = crate::scene::LayoutBox {
                        x: bounds.x + current_state.offset[0],
                        y: bounds.y + current_state.offset[1],
                        width: bounds.width,
                        height: bounds.height,
                    };
                    let Some(clipped_rect) = clip_rect(rect, current_state.clip) else {
                        continue;
                    };
                    self.quad_instances.push(QuadInstance {
                        rect: [
                            clipped_rect.x * scale_factor,
                            clipped_rect.y * scale_factor,
                            clipped_rect.width * scale_factor,
                            clipped_rect.height * scale_factor,
                        ],
                        color: [color.r, color.g, color.b, color.a * current_state.opacity],
                    });
                    self.push_gpu_batch(
                        PipelineKey::Quad,
                        TextureBindingKey::None,
                        batch.clip_class,
                        current_state.clip,
                        batch.effect_class,
                        start..self.quad_instances.len() as u32,
                    );
                }
                Primitive::Text {
                    bounds,
                    layout,
                    color,
                } => {
                    debug_assert_eq!(batch.material_class, MaterialClass::Text);
                    self.collect_text_primitive_instances(
                        text_system,
                        scale_factor,
                        TextPrimitiveParams {
                            bounds,
                            layout,
                            color,
                            scene_state: current_state,
                            batch,
                        },
                    );
                }
            }
        }

        let mut child = node.first_child;
        while let Some(child_id) = child {
            self.collect_node_instances(
                scene,
                text_system,
                scale_factor,
                child_id,
                current_state,
                batch_cursor,
            );
            child = scene.scene_nodes[child_id.0 as usize].next_sibling;
        }
    }

    fn collect_text_primitive_instances(
        &mut self,
        text_system: &mut TextSystem,
        scale_factor: f32,
        params: TextPrimitiveParams<'_>,
    ) {
        let mut active_batch = None;

        for run in &params.layout.runs {
            for glyph in &run.glyphs {
                let physical = glyph.physical(
                    (
                        (params.bounds.x + params.scene_state.offset[0]) * scale_factor,
                        (params.bounds.y + params.scene_state.offset[1] + run.baseline)
                            * scale_factor,
                    ),
                    scale_factor,
                );
                let Some((atlas_kind, entry)) =
                    self.ensure_glyph_entry(text_system, physical.cache_key)
                else {
                    continue;
                };
                let rect = crate::scene::LayoutBox {
                    x: (physical.x + entry.placement_left) as f32,
                    y: (physical.y - entry.placement_top) as f32,
                    width: entry.width as f32,
                    height: entry.height as f32,
                };
                let uv = crate::scene::LayoutBox {
                    x: entry.uv_rect[0],
                    y: entry.uv_rect[1],
                    width: entry.uv_rect[2],
                    height: entry.uv_rect[3],
                };
                let Some((clipped_rect, clipped_uv)) =
                    clip_text_glyph(rect, uv, params.scene_state.clip)
                else {
                    continue;
                };
                let rect = [
                    clipped_rect.x,
                    clipped_rect.y,
                    clipped_rect.width,
                    clipped_rect.height,
                ];
                let glyph_color = glyph
                    .color_opt
                    .map(cosmic_to_style_color)
                    .unwrap_or(*params.color);

                match atlas_kind {
                    GlyphAtlasKind::Mono => {
                        self.start_or_switch_batch(
                            &mut active_batch,
                            PipelineKey::MonoText,
                            TextureBindingKey::MonoGlyphAtlas(entry.page_id),
                            params.batch.clip_class,
                            params.scene_state.clip,
                            params.batch.effect_class,
                        );
                        self.mono_text_instances.push(TextInstance {
                            rect,
                            uv_rect: [
                                clipped_uv.x,
                                clipped_uv.y,
                                clipped_uv.width,
                                clipped_uv.height,
                            ],
                            color: [
                                glyph_color.r,
                                glyph_color.g,
                                glyph_color.b,
                                glyph_color.a * params.scene_state.opacity,
                            ],
                        });
                    }
                    GlyphAtlasKind::Color => {
                        self.start_or_switch_batch(
                            &mut active_batch,
                            PipelineKey::ColorText,
                            TextureBindingKey::ColorGlyphAtlas(entry.page_id),
                            params.batch.clip_class,
                            params.scene_state.clip,
                            params.batch.effect_class,
                        );
                        self.color_text_instances.push(ColorTextInstance {
                            rect,
                            uv_rect: [
                                clipped_uv.x,
                                clipped_uv.y,
                                clipped_uv.width,
                                clipped_uv.height,
                            ],
                            alpha: glyph_color.a * params.scene_state.opacity,
                        });
                    }
                }
            }
        }

        self.finish_active_batch(
            &mut active_batch,
            params.batch.clip_class,
            params.scene_state.clip,
            params.batch.effect_class,
        );
    }
}

fn push_gpu_batch(batches: &mut Vec<GpuBatch>, batch: GpuBatch) {
    let mut builder = GpuBatchBuilder {
        batches: std::mem::take(batches),
    };
    builder.push(batch);
    *batches = builder.batches;
}

fn effect_render_policy(effect_policy: BatchEffectPolicy) -> EffectRenderPolicy {
    match effect_policy {
        BatchEffectPolicy::None => EffectRenderPolicy::Direct,
        BatchEffectPolicy::Opacity => EffectRenderPolicy::InlineOpacity,
    }
}

fn can_merge_gpu_batches(previous: &GpuBatch, next: &GpuBatch) -> bool {
    previous.pipeline_key == next.pipeline_key
        && previous.texture_binding == next.texture_binding
        && previous.clip_class == next.clip_class
        && previous.clip_bounds == next.clip_bounds
        && previous.effect_class == next.effect_class
        && previous.instance_range.end == next.instance_range.start
}

fn clip_bounds_to_scissor_rect(
    clip_bounds: Option<crate::scene::LayoutBox>,
    scale_factor: f32,
    viewport_width: u32,
    viewport_height: u32,
) -> Option<ScissorRect> {
    let clip_bounds = clip_bounds?;
    let scale_factor = scale_factor.max(f32::MIN_POSITIVE);
    let left = (clip_bounds.x * scale_factor).floor().max(0.0);
    let top = (clip_bounds.y * scale_factor).floor().max(0.0);
    let right = ((clip_bounds.x + clip_bounds.width) * scale_factor)
        .ceil()
        .min(viewport_width as f32);
    let bottom = ((clip_bounds.y + clip_bounds.height) * scale_factor)
        .ceil()
        .min(viewport_height as f32);

    if right <= left || bottom <= top {
        return None;
    }

    Some(ScissorRect {
        x: left as u32,
        y: top as u32,
        width: (right - left) as u32,
        height: (bottom - top) as u32,
    })
}

fn draw_gpu_batch(pass: &mut wgpu::RenderPass<'_>, instance_range: Range<u32>) {
    pass.draw(0..6, instance_range);
}

fn draw_gpu_batch_inline_opacity(pass: &mut wgpu::RenderPass<'_>, instance_range: Range<u32>) {
    pass.draw(0..6, instance_range);
}

fn clip_rect(
    rect: crate::scene::LayoutBox,
    clip: Option<crate::scene::LayoutBox>,
) -> Option<crate::scene::LayoutBox> {
    clip.map_or(Some(rect), |clip| intersect_rect(rect, clip))
}

fn clip_text_glyph(
    rect: crate::scene::LayoutBox,
    uv: crate::scene::LayoutBox,
    clip: Option<crate::scene::LayoutBox>,
) -> Option<(crate::scene::LayoutBox, crate::scene::LayoutBox)> {
    let clipped = clip_rect(rect, clip)?;
    if rect.width <= 0.0 || rect.height <= 0.0 {
        return None;
    }

    let left_ratio = (clipped.x - rect.x) / rect.width;
    let top_ratio = (clipped.y - rect.y) / rect.height;
    let right_ratio = (clipped.x + clipped.width - rect.x) / rect.width;
    let bottom_ratio = (clipped.y + clipped.height - rect.y) / rect.height;

    let clipped_uv = crate::scene::LayoutBox {
        x: uv.x + uv.width * left_ratio,
        y: uv.y + uv.height * top_ratio,
        width: uv.width * (right_ratio - left_ratio),
        height: uv.height * (bottom_ratio - top_ratio),
    };

    Some((clipped, clipped_uv))
}

fn intersect_clip(
    a: Option<crate::scene::LayoutBox>,
    b: Option<crate::scene::LayoutBox>,
) -> Option<crate::scene::LayoutBox> {
    match (a, b) {
        (Some(a), Some(b)) => intersect_rect(a, b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn intersect_rect(
    a: crate::scene::LayoutBox,
    b: crate::scene::LayoutBox,
) -> Option<crate::scene::LayoutBox> {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);

    if right <= left || bottom <= top {
        return None;
    }

    Some(crate::scene::LayoutBox {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn create_quad_pipeline(
    device: &Device,
    view_layout: &BindGroupLayout,
    surface_format: TextureFormat,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("nekoui_quad_shader"),
        source: ShaderSource::Wgsl(QUAD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("nekoui_quad_pipeline_layout"),
        bind_group_layouts: &[Some(view_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("nekoui_quad_pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadInstance>() as u64,
                step_mode: VertexStepMode::Instance,
                attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4],
            }],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_text_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("nekoui_text_texture_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
        ],
    })
}

fn create_mono_text_pipeline(
    device: &Device,
    view_layout: &BindGroupLayout,
    glyph_layout: &BindGroupLayout,
    surface_format: TextureFormat,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("nekoui_text_shader"),
        source: ShaderSource::Wgsl(TEXT_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("nekoui_mono_text_pipeline_layout"),
        bind_group_layouts: &[Some(view_layout), Some(glyph_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("nekoui_mono_text_pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_mono"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<TextInstance>() as u64,
                step_mode: VertexStepMode::Instance,
                attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4],
            }],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_mono"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_color_text_pipeline(
    device: &Device,
    view_layout: &BindGroupLayout,
    glyph_layout: &BindGroupLayout,
    surface_format: TextureFormat,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("nekoui_text_shader"),
        source: ShaderSource::Wgsl(TEXT_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("nekoui_color_text_pipeline_layout"),
        bind_group_layouts: &[Some(view_layout), Some(glyph_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("nekoui_color_text_pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_color"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<ColorTextInstance>() as u64,
                step_mode: VertexStepMode::Instance,
                attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32],
            }],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_color"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_instance_buffer<T: Pod>(device: &Device, label: &str, capacity: usize) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (std::mem::size_of::<T>() * capacity.max(1)) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn cosmic_to_style_color(color: CosmicColor) -> Color {
    Color::rgba(
        f32::from(color.r()) / 255.0,
        f32::from(color.g()) / 255.0,
        f32::from(color.b()) / 255.0,
        f32::from(color.a()) / 255.0,
    )
}

fn color_to_wgpu(color: Color) -> wgpu::Color {
    wgpu::Color {
        r: color.r as f64,
        g: color.g as f64,
        b: color.b as f64,
        a: color.a as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BatchClipPolicy, BatchEffectPolicy, BatchSubmitState, EffectRenderPolicy, GpuBatch,
        PipelineKey, ScissorRect, TextureBindingKey, can_merge_gpu_batches,
        clip_bounds_to_scissor_rect, effect_render_policy, push_gpu_batch,
    };
    use crate::scene::{ClipClass, EffectClass, LayoutBox};

    #[test]
    fn gpu_batch_merge_respects_clip_and_effect_boundaries() {
        let mut batches = Vec::new();
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::MonoText,
                texture_binding: TextureBindingKey::MonoGlyphAtlas(0),
                clip_class: ClipClass::None,
                clip_bounds: None,
                effect_class: EffectClass::None,
                instance_range: 0..4,
            },
        );
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::MonoText,
                texture_binding: TextureBindingKey::MonoGlyphAtlas(0),
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 10.0,
                    y: 20.0,
                    width: 40.0,
                    height: 50.0,
                }),
                effect_class: EffectClass::None,
                instance_range: 4..8,
            },
        );
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::MonoText,
                texture_binding: TextureBindingKey::MonoGlyphAtlas(0),
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 10.0,
                    y: 20.0,
                    width: 40.0,
                    height: 50.0,
                }),
                effect_class: EffectClass::Opacity,
                instance_range: 8..12,
            },
        );

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].instance_range, 0..4);
        assert_eq!(batches[1].instance_range, 4..8);
        assert_eq!(batches[2].instance_range, 8..12);
        assert!(!can_merge_gpu_batches(&batches[0], &batches[1]));
        assert!(!can_merge_gpu_batches(&batches[1], &batches[2]));
    }

    #[test]
    fn gpu_batch_merge_coalesces_compatible_adjacent_ranges() {
        let mut batches = Vec::new();
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::Quad,
                texture_binding: TextureBindingKey::None,
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 4.0,
                    y: 8.0,
                    width: 16.0,
                    height: 12.0,
                }),
                effect_class: EffectClass::Opacity,
                instance_range: 0..2,
            },
        );
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::Quad,
                texture_binding: TextureBindingKey::None,
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 4.0,
                    y: 8.0,
                    width: 16.0,
                    height: 12.0,
                }),
                effect_class: EffectClass::Opacity,
                instance_range: 2..5,
            },
        );

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instance_range, 0..5);
    }

    #[test]
    fn batch_submit_state_tracks_clip_and_effect_policy() {
        let batch = GpuBatch {
            pipeline_key: PipelineKey::ColorText,
            texture_binding: TextureBindingKey::ColorGlyphAtlas(3),
            clip_class: ClipClass::Rect,
            clip_bounds: Some(LayoutBox {
                x: 1.0,
                y: 2.0,
                width: 30.0,
                height: 40.0,
            }),
            effect_class: EffectClass::Opacity,
            instance_range: 3..9,
        };

        let submit_state = BatchSubmitState::from(&batch);
        assert_eq!(submit_state.pipeline_key, PipelineKey::ColorText);
        assert_eq!(
            submit_state.texture_binding,
            TextureBindingKey::ColorGlyphAtlas(3)
        );
        assert_eq!(submit_state.clip_policy, BatchClipPolicy::Rect);
        assert_eq!(
            submit_state.clip_bounds,
            Some(LayoutBox {
                x: 1.0,
                y: 2.0,
                width: 30.0,
                height: 40.0,
            })
        );
        assert_eq!(submit_state.effect_policy, BatchEffectPolicy::Opacity);
        assert_eq!(
            submit_state.effect_render_policy,
            EffectRenderPolicy::InlineOpacity
        );
    }

    #[test]
    fn gpu_batch_merge_respects_clip_bounds() {
        let mut batches = Vec::new();
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::Quad,
                texture_binding: TextureBindingKey::None,
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                }),
                effect_class: EffectClass::None,
                instance_range: 0..2,
            },
        );
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::Quad,
                texture_binding: TextureBindingKey::None,
                clip_class: ClipClass::Rect,
                clip_bounds: Some(LayoutBox {
                    x: 2.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                }),
                effect_class: EffectClass::None,
                instance_range: 2..4,
            },
        );

        assert_eq!(batches.len(), 2);
        assert!(!can_merge_gpu_batches(&batches[0], &batches[1]));
    }

    #[test]
    fn gpu_batch_merge_respects_atlas_page_boundaries() {
        let mut batches = Vec::new();
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::MonoText,
                texture_binding: TextureBindingKey::MonoGlyphAtlas(1),
                clip_class: ClipClass::None,
                clip_bounds: None,
                effect_class: EffectClass::None,
                instance_range: 0..2,
            },
        );
        push_gpu_batch(
            &mut batches,
            GpuBatch {
                pipeline_key: PipelineKey::MonoText,
                texture_binding: TextureBindingKey::MonoGlyphAtlas(2),
                clip_class: ClipClass::None,
                clip_bounds: None,
                effect_class: EffectClass::None,
                instance_range: 2..4,
            },
        );

        assert_eq!(batches.len(), 2);
        assert!(!can_merge_gpu_batches(&batches[0], &batches[1]));
    }

    #[test]
    fn clip_bounds_to_scissor_rect_clamps_to_viewport() {
        let scissor = clip_bounds_to_scissor_rect(
            Some(LayoutBox {
                x: -4.25,
                y: 2.25,
                width: 20.75,
                height: 40.5,
            }),
            2.0,
            24,
            32,
        )
        .unwrap();

        assert_eq!(
            scissor,
            ScissorRect {
                x: 0,
                y: 4,
                width: 24,
                height: 28,
            }
        );
    }

    #[test]
    fn effect_policy_maps_to_real_render_policy() {
        assert_eq!(
            effect_render_policy(BatchEffectPolicy::None),
            EffectRenderPolicy::Direct
        );
        assert_eq!(
            effect_render_policy(BatchEffectPolicy::Opacity),
            EffectRenderPolicy::InlineOpacity
        );
    }
}
