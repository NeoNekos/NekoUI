use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::scene::{ClipClass, EffectClass, LogicalBatch};
use crate::style::Color;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ViewUniform {
    pub(super) viewport: [f32; 2],
    pub(super) _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct RectInstance {
    pub(super) rect: [f32; 4],
    pub(super) fill_start_color: [f32; 4],
    pub(super) fill_end_color: [f32; 4],
    pub(super) fill_meta: [f32; 4],
    pub(super) corner_radii: [f32; 4],
    pub(super) border_widths: [f32; 4],
    pub(super) border_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct TextInstance {
    pub(super) rect: [f32; 4],
    pub(super) uv_rect: [f32; 4],
    pub(super) color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct ColorTextInstance {
    pub(super) rect: [f32; 4],
    pub(super) uv_rect: [f32; 4],
    pub(super) alpha: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreparedFrameKey {
    pub(super) scene_nodes_ptr: usize,
    pub(super) scene_nodes_len: usize,
    pub(super) primitives_ptr: usize,
    pub(super) primitives_len: usize,
    pub(super) logical_batches_ptr: usize,
    pub(super) logical_batches_len: usize,
    pub(super) scale_factor_bits: u32,
    pub(super) mono_atlas_generation: u64,
    pub(super) color_atlas_generation: u64,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedFrame {
    pub(super) key: PreparedFrameKey,
    pub(super) rect_instances: Arc<Vec<RectInstance>>,
    pub(super) mono_text_instances: Arc<Vec<TextInstance>>,
    pub(super) color_text_instances: Arc<Vec<ColorTextInstance>>,
    pub(super) gpu_batches: Arc<Vec<GpuBatch>>,
    pub(super) uploaded_buffer_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PipelineKey {
    Rect,
    MonoText,
    ColorText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TextureBindingKey {
    None,
    MonoGlyphAtlas(u32),
    ColorGlyphAtlas(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveBatch {
    pub(super) pipeline_key: PipelineKey,
    pub(super) texture_binding: TextureBindingKey,
    pub(super) start: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct GpuBatch {
    pub(super) pipeline_key: PipelineKey,
    pub(super) texture_binding: TextureBindingKey,
    pub(super) clip_class: ClipClass,
    pub(super) clip_bounds: Option<crate::scene::LayoutBox>,
    pub(super) effect_class: EffectClass,
    pub(super) instance_range: Range<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchClipPolicy {
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
pub(super) enum BatchEffectPolicy {
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
pub(super) enum EffectRenderPolicy {
    Direct,
    InlineOpacity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BatchSubmitState {
    pub(super) pipeline_key: PipelineKey,
    pub(super) texture_binding: TextureBindingKey,
    pub(super) clip_policy: BatchClipPolicy,
    pub(super) clip_bounds: Option<crate::scene::LayoutBox>,
    pub(super) effect_policy: BatchEffectPolicy,
    pub(super) effect_render_policy: EffectRenderPolicy,
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
            effect_render_policy: super::submit::effect_render_policy(effect_policy),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScissorRect {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Default)]
pub(super) struct GpuBatchBuilder {
    pub(super) batches: Vec<GpuBatch>,
}

impl GpuBatchBuilder {
    pub(super) fn push(&mut self, batch: GpuBatch) {
        if batch.instance_range.is_empty() {
            return;
        }

        if let Some(previous) = self.batches.last_mut()
            && super::submit::can_merge_gpu_batches(previous, &batch)
        {
            previous.instance_range.end = batch.instance_range.end;
            return;
        }

        self.batches.push(batch);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SceneWalkState {
    pub(super) offset: [f32; 2],
    pub(super) opacity: f32,
    pub(super) clip: Option<crate::scene::LayoutBox>,
}

pub(super) struct TextPrimitiveParams<'a> {
    pub(super) bounds: &'a crate::scene::LayoutBox,
    pub(super) layout: &'a crate::text_system::TextLayout,
    pub(super) color: &'a Color,
    pub(super) scene_state: SceneWalkState,
    pub(super) batch: &'a LogicalBatch,
}

pub(super) struct LogicalBatchCursor<'a> {
    batches: &'a [LogicalBatch],
    index: usize,
}

impl<'a> LogicalBatchCursor<'a> {
    pub(super) fn new(batches: &'a [LogicalBatch]) -> Self {
        Self { batches, index: 0 }
    }

    pub(super) fn batch_for_primitive(&mut self, primitive_index: u32) -> &'a LogicalBatch {
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
    pub(super) surface: wgpu::Surface<'static>,
    pub(super) config: wgpu::SurfaceConfiguration,
    pub(super) current_size: crate::window::WindowSize,
    pub(super) suspended: bool,
    pub(super) prepared_frame: Option<PreparedFrame>,
}
