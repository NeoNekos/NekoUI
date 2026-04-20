use std::sync::Arc;

use cosmic_text::{CacheKey, SwashContent};

use crate::platform::wgpu::atlas::{AtlasEntry, GlyphAtlasKind};
use crate::scene::{CompiledScene, MaterialClass, Primitive, SceneNodeId};
use crate::text_system::TextSystem;

use super::{
    RenderSystem,
    submit::push_gpu_batch,
    types::{
        ActiveBatch, ColorTextInstance, GpuBatch, LogicalBatchCursor, PipelineKey, PreparedFrame,
        PreparedFrameKey, RectInstance, SceneWalkState, TextInstance, TextPrimitiveParams,
        TextureBindingKey,
    },
};

impl RenderSystem {
    pub(super) fn prepare_frame(
        &mut self,
        state: &mut super::types::WindowRenderState,
        scene: &CompiledScene,
        text_system: &mut TextSystem,
        scale_factor: f32,
    ) {
        self.mono_atlas.begin_frame();
        self.color_atlas.begin_frame();

        let current_key = self.prepared_frame_key(scene, scale_factor);
        if state
            .prepared_frame
            .as_ref()
            .is_some_and(|prepared| prepared.key == current_key)
        {
            return;
        }

        self.rect_instances.clear();
        self.mono_text_instances.clear();
        self.color_text_instances.clear();
        self.gpu_batches.clear();
        self.collect_instances(scene, text_system, scale_factor);
        let final_key = self.prepared_frame_key(scene, scale_factor);
        state.prepared_frame = Some(PreparedFrame {
            key: final_key,
            rect_instances: Arc::new(std::mem::take(&mut self.rect_instances)),
            mono_text_instances: Arc::new(std::mem::take(&mut self.mono_text_instances)),
            color_text_instances: Arc::new(std::mem::take(&mut self.color_text_instances)),
            gpu_batches: Arc::new(std::mem::take(&mut self.gpu_batches)),
            uploaded_buffer_epoch: 0,
        });
    }

    pub(super) fn prepared_frame_key(
        &self,
        scene: &CompiledScene,
        scale_factor: f32,
    ) -> PreparedFrameKey {
        PreparedFrameKey {
            scene_nodes_ptr: scene.scene_nodes.as_ptr() as usize,
            scene_nodes_len: scene.scene_nodes.len(),
            primitives_ptr: scene.primitives.as_ptr() as usize,
            primitives_len: scene.primitives.len(),
            logical_batches_ptr: scene.logical_batches.as_ptr() as usize,
            logical_batches_len: scene.logical_batches.len(),
            scale_factor_bits: scale_factor.to_bits(),
            mono_atlas_generation: self.mono_atlas.generation(),
            color_atlas_generation: self.color_atlas.generation(),
        }
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

    pub(super) fn ensure_glyph_entry(
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
            SwashContent::Color => self
                .color_atlas
                .upload_color(&self.context.device, &self.context.queue, cache_key, &image)
                .map(|entry| (GlyphAtlasKind::Color, entry)),
            SwashContent::Mask | SwashContent::SubpixelMask => self
                .mono_atlas
                .upload_mask(&self.context.device, &self.context.queue, cache_key, &image)
                .map(|entry| (GlyphAtlasKind::Mono, entry)),
        }
    }

    fn instance_count_for(&self, pipeline_key: PipelineKey) -> u32 {
        match pipeline_key {
            PipelineKey::Rect => self.rect_instances.len() as u32,
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
            clip: super::prepare::intersect_clip(parent_state.clip, local_clip),
        };

        for primitive_index in node.primitive_range.as_range() {
            let batch = batch_cursor.batch_for_primitive(primitive_index as u32);
            match &scene.primitives[primitive_index] {
                Primitive::Rect(rect_primitive) => {
                    debug_assert_eq!(batch.material_class, MaterialClass::Rect);
                    let start = self.rect_instances.len() as u32;
                    let (fill_start_color, fill_end_color, fill_meta) = match rect_primitive.fill {
                        crate::scene::RectFill::Solid(color) => {
                            (color, color, [0.0, 0.0, 0.0, 0.0])
                        }
                        crate::scene::RectFill::LinearGradient(gradient) => (
                            gradient.start_color,
                            gradient.end_color,
                            [1.0, gradient.angle_radians, 0.0, 0.0],
                        ),
                    };
                    let rect_bounds = crate::scene::LayoutBox {
                        x: rect_primitive.bounds.x + current_state.offset[0],
                        y: rect_primitive.bounds.y + current_state.offset[1],
                        width: rect_primitive.bounds.width,
                        height: rect_primitive.bounds.height,
                    };
                    let Some(clipped_rect) = clip_rect(rect_bounds, current_state.clip) else {
                        continue;
                    };
                    self.rect_instances.push(RectInstance {
                        rect: [
                            clipped_rect.x * scale_factor,
                            clipped_rect.y * scale_factor,
                            clipped_rect.width * scale_factor,
                            clipped_rect.height * scale_factor,
                        ],
                        fill_start_color: [
                            fill_start_color.r,
                            fill_start_color.g,
                            fill_start_color.b,
                            fill_start_color.a * current_state.opacity,
                        ],
                        fill_end_color: [
                            fill_end_color.r,
                            fill_end_color.g,
                            fill_end_color.b,
                            fill_end_color.a * current_state.opacity,
                        ],
                        fill_meta,
                        corner_radii: [
                            rect_primitive.corner_radii.top_left * scale_factor,
                            rect_primitive.corner_radii.top_right * scale_factor,
                            rect_primitive.corner_radii.bottom_right * scale_factor,
                            rect_primitive.corner_radii.bottom_left * scale_factor,
                        ],
                        border_widths: [
                            rect_primitive.border_widths.top * scale_factor,
                            rect_primitive.border_widths.right * scale_factor,
                            rect_primitive.border_widths.bottom * scale_factor,
                            rect_primitive.border_widths.left * scale_factor,
                        ],
                        border_color: rect_primitive
                            .border_color
                            .map(|border_color| {
                                [
                                    border_color.r,
                                    border_color.g,
                                    border_color.b,
                                    border_color.a * current_state.opacity,
                                ]
                            })
                            .unwrap_or([0.0; 4]),
                    });
                    self.push_gpu_batch(
                        PipelineKey::Rect,
                        TextureBindingKey::None,
                        batch.clip_class,
                        current_state.clip,
                        batch.effect_class,
                        start..self.rect_instances.len() as u32,
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
                    .map(super::cosmic_to_style_color)
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

    fn start_or_switch_batch(
        &mut self,
        active_batch: &mut Option<ActiveBatch>,
        pipeline_key: super::types::PipelineKey,
        texture_binding: super::types::TextureBindingKey,
        clip_class: crate::scene::ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: crate::scene::EffectClass,
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
        clip_class: crate::scene::ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: crate::scene::EffectClass,
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
        clip_class: crate::scene::ClipClass,
        clip_bounds: Option<crate::scene::LayoutBox>,
        effect_class: crate::scene::EffectClass,
        instance_range: std::ops::Range<u32>,
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
}

pub(super) fn clip_rect(
    rect: crate::scene::LayoutBox,
    clip: Option<crate::scene::LayoutBox>,
) -> Option<crate::scene::LayoutBox> {
    clip.map_or(Some(rect), |clip| intersect_rect(rect, clip))
}

pub(super) fn clip_text_glyph(
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

pub(super) fn intersect_clip(
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

pub(super) fn intersect_rect(
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
