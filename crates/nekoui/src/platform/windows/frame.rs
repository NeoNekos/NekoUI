use std::time::{Duration, Instant};

use crate::error::{ErrorKind, NekoError, NekoResult};
use crate::platform::{
    BackendFrameReceipt, BackendFrameStatus, BackendSurfaceState, Renderability,
};
use crate::render::{DrawItemKind, PreparedFrame};
use crate::text::FontManager;
use crate::window::AnyWindowHandle;

use super::box_shape::{box_shape_count, collect_box_shapes, unsupported_draw_items};
use super::clip::ClipStack;
use super::device::D3d11DeviceState;
use super::glyph::{
    GlyphAtlas, GlyphDrawFormat, GlyphDrawPlan, GlyphUnsupportedReport, prepare_glyph_atlases,
};
use super::glyph_pipeline::{D3d11GlyphColorPipeline, D3d11GlyphMonoPipeline};
use super::pipeline::D3d11BoxShapePipeline;
use super::surface::DxgiSurface;

pub(super) struct FrameRecordResources<'a> {
    pub(super) font_manager: &'a FontManager,
    pub(super) mono_glyph_atlas: Option<&'a mut GlyphAtlas>,
    pub(super) color_glyph_atlas: &'a mut Option<GlyphAtlas>,
    pub(super) mono_glyph_pipeline: &'a mut Option<D3d11GlyphMonoPipeline>,
    pub(super) color_glyph_pipeline: &'a mut Option<D3d11GlyphColorPipeline>,
    pub(super) glyph_draw_plan: &'a mut GlyphDrawPlan,
    pub(super) box_shape_pipeline: Option<&'a mut D3d11BoxShapePipeline>,
}

#[derive(Clone)]
pub(super) struct FrameReport {
    pub(super) receipt: BackendFrameReceipt,
    pub(super) duration: Duration,
    pub(super) error: Option<NekoError>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FrameRecordReport {
    pub(super) glyph_unsupported: GlyphUnsupportedReport,
    pub(super) mono_glyph_pipeline_created: bool,
    pub(super) color_glyph_pipeline_created: bool,
}

pub(super) struct ActiveFrame {
    window: AnyWindowHandle,
    generation: u64,
    unsupported_draw_items: usize,
    started: Instant,
    phase: FramePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FramePhase {
    Begun,
    Recorded,
}

impl ActiveFrame {
    pub(super) fn begin(
        window: AnyWindowHandle,
        surface: &DxgiSurface,
        prepared: &PreparedFrame,
        renderability: Renderability,
    ) -> Result<Self, BackendFrameReceipt> {
        let generation = prepared.generation().surface_generation().unwrap_or(0);
        if !renderability.is_renderable() || surface.state() != BackendSurfaceState::Ready {
            return Err(receipt(
                window,
                generation,
                BackendFrameStatus::NotRenderable,
                surface.state(),
                0,
                0,
                "surface is not renderable",
            ));
        }
        if generation == 0 || generation != surface.generation() {
            return Err(receipt(
                window,
                generation,
                BackendFrameStatus::StaleDropped,
                surface.state(),
                0,
                1,
                "prepared frame surface generation is stale",
            ));
        }
        Ok(Self {
            window,
            generation,
            unsupported_draw_items: 0,
            started: Instant::now(),
            phase: FramePhase::Begun,
        })
    }

    pub(super) fn record(
        &mut self,
        device: &D3d11DeviceState,
        surface: &DxgiSurface,
        prepared: &PreparedFrame,
        resources: FrameRecordResources<'_>,
    ) -> NekoResult<FrameRecordReport> {
        if self.generation != surface.generation()
            || prepared.generation().surface_generation() != Some(surface.generation())
        {
            return Err(NekoError::stale("active frame generation is stale"));
        }
        let mut glyph_unsupported = GlyphUnsupportedReport::default();
        self.unsupported_draw_items = unsupported_draw_items(prepared);
        surface.clear(device, super::surface::WINDOWS_BACKEND_CLEAR_COLOR)?;
        let mono_glyph_pipeline = resources.mono_glyph_pipeline;
        let color_glyph_pipeline = resources.color_glyph_pipeline;
        let color_glyph_atlas = resources.color_glyph_atlas;
        let glyph_draw_plan = resources.glyph_draw_plan;
        let mut mono_glyph_pipeline_created = false;
        let mut color_glyph_pipeline_created = false;
        if has_supported_glyph_text(prepared) {
            let mono_atlas = resources.mono_glyph_atlas.ok_or_else(|| {
                NekoError::unsupported("D3D11 mono glyph atlas is unavailable for text draw items")
            })?;
            glyph_unsupported.add(prepare_glyph_atlases(
                prepared,
                mono_atlas,
                color_glyph_atlas,
                resources.font_manager,
            )?);
            glyph_unsupported.add(super::glyph::collect_glyph_draw_plan(
                prepared,
                mono_atlas,
                color_glyph_atlas.as_ref(),
                glyph_draw_plan,
            )?);
            if glyph_draw_plan.has_mono_draws() {
                if mono_glyph_pipeline.is_none() {
                    *mono_glyph_pipeline = Some(D3d11GlyphMonoPipeline::new(device)?);
                    mono_glyph_pipeline_created = true;
                }
                let mono_pipeline = mono_glyph_pipeline.as_mut().ok_or_else(|| {
                    NekoError::unsupported("D3D11 mono glyph pipeline is unavailable")
                })?;
                mono_pipeline.upload_if_dirty(device, mono_atlas)?;
            }
            if glyph_draw_plan.has_color_draws() {
                let color_atlas = color_glyph_atlas.as_mut().ok_or_else(|| {
                    NekoError::unsupported(
                        "D3D11 color glyph atlas is unavailable for color glyph draw items",
                    )
                })?;
                if color_glyph_pipeline.is_none() {
                    *color_glyph_pipeline = Some(D3d11GlyphColorPipeline::new(device)?);
                    color_glyph_pipeline_created = true;
                }
                let color_pipeline = color_glyph_pipeline.as_mut().ok_or_else(|| {
                    NekoError::unsupported("D3D11 color glyph pipeline is unavailable")
                })?;
                color_pipeline.upload_if_dirty(device, color_atlas)?;
            }
        } else {
            glyph_draw_plan.clear();
        };
        self.unsupported_draw_items += glyph_unsupported.skipped_glyph_instances();
        let shapes = collect_box_shapes(prepared);
        let mut clip_stack = ClipStack::default();
        let mut shape_index = 0;
        let mut glyph_run_index = 0;
        let mut box_shape_pipeline = resources.box_shape_pipeline;
        for item in prepared.draw_items() {
            match item.kind() {
                DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
                DrawItemKind::ClipPop => clip_stack.pop(),
                DrawItemKind::BoxShape { shape }
                    if super::box_shape::supported_box_shape(*shape) =>
                {
                    let end = shape_index + 1;
                    if let Some(scissor) = clip_stack.active_scissor(prepared.context()) {
                        let pipeline = box_shape_pipeline.as_deref_mut().ok_or_else(|| {
                            NekoError::unsupported(
                                "D3D11 box shape pipeline is unavailable without generated framework shader artifacts",
                            )
                        })?;
                        pipeline.draw(
                            device,
                            surface.render_target_view()?,
                            prepared.context(),
                            &shapes[shape_index..end],
                            scissor,
                        )?;
                    }
                    shape_index = end;
                }
                DrawItemKind::Rect { color }
                    if color.to_current_backend_sdr_srgb_rgba().is_some() =>
                {
                    let end = shape_index + 1;
                    if let Some(scissor) = clip_stack.active_scissor(prepared.context()) {
                        let pipeline = box_shape_pipeline.as_deref_mut().ok_or_else(|| {
                            NekoError::unsupported(
                                "D3D11 box shape pipeline is unavailable without generated framework shader artifacts",
                            )
                        })?;
                        pipeline.draw(
                            device,
                            surface.render_target_view()?,
                            prepared.context(),
                            &shapes[shape_index..end],
                            scissor,
                        )?;
                    }
                    shape_index = end;
                }
                _ if item.kind().supported_windows_glyph_text() => {
                    let scissor = clip_stack.active_scissor(prepared.context());
                    while glyph_run_index < glyph_draw_plan.runs().len()
                        && glyph_draw_plan.runs()[glyph_run_index].order == item.order()
                    {
                        if let Some(scissor) = scissor {
                            let run = &glyph_draw_plan.runs()[glyph_run_index];
                            match run.format {
                                GlyphDrawFormat::MonoMask => {
                                    let pipeline = mono_glyph_pipeline.as_mut().ok_or_else(|| {
                                        NekoError::unsupported(
                                            "D3D11 mono glyph pipeline is unavailable without generated framework shader artifacts",
                                        )
                                    })?;
                                    pipeline.draw_glyphs(
                                        device,
                                        surface.render_target_view()?,
                                        prepared.context(),
                                        &glyph_draw_plan.mono_draws()[run.range.clone()],
                                        scissor,
                                    )?;
                                }
                                GlyphDrawFormat::ColorRgba => {
                                    let pipeline = color_glyph_pipeline.as_mut().ok_or_else(|| {
                                        NekoError::unsupported(
                                            "D3D11 color glyph pipeline is unavailable without generated framework shader artifacts",
                                        )
                                    })?;
                                    pipeline.draw_glyphs(
                                        device,
                                        surface.render_target_view()?,
                                        prepared.context(),
                                        &glyph_draw_plan.color_draws()[run.range.clone()],
                                        scissor,
                                    )?;
                                }
                            }
                        }
                        glyph_run_index += 1;
                    }
                }
                _ => {}
            }
        }
        debug_assert!(
            clip_stack.is_empty(),
            "prepared draw items ended with an unbalanced clip stack"
        );
        self.phase = FramePhase::Recorded;
        Ok(FrameRecordReport {
            glyph_unsupported,
            mono_glyph_pipeline_created,
            color_glyph_pipeline_created,
        })
    }

    pub(super) fn present(self, surface: &DxgiSurface) -> FrameReport {
        if self.phase != FramePhase::Recorded || self.generation != surface.generation() {
            return FrameReport {
                duration: self.started.elapsed(),
                receipt: receipt(
                    self.window,
                    self.generation,
                    BackendFrameStatus::StaleDropped,
                    surface.state(),
                    self.unsupported_draw_items,
                    1,
                    "active frame was stale before present",
                ),
                error: None,
            };
        }
        match surface.present() {
            Ok(()) => FrameReport {
                duration: self.started.elapsed(),
                receipt: receipt(
                    self.window,
                    self.generation,
                    BackendFrameStatus::Presented,
                    surface.state(),
                    self.unsupported_draw_items,
                    0,
                    "frame cleared and presented",
                ),
                error: None,
            },
            Err(error) => self.fail(surface.state(), "active frame failed during present", error),
        }
    }

    pub(super) fn abort(
        self,
        surface_state: BackendSurfaceState,
        message: &'static str,
    ) -> FrameReport {
        FrameReport {
            duration: self.started.elapsed(),
            receipt: receipt(
                self.window,
                self.generation,
                BackendFrameStatus::Aborted,
                surface_state,
                self.unsupported_draw_items,
                0,
                message,
            ),
            error: None,
        }
    }

    pub(super) fn fail(
        self,
        surface_state: BackendSurfaceState,
        message: &'static str,
        error: NekoError,
    ) -> FrameReport {
        let failure_kind = error.kind();
        FrameReport {
            duration: self.started.elapsed(),
            receipt: build_receipt(ReceiptParts {
                window: self.window,
                surface_generation: self.generation,
                status: BackendFrameStatus::Failed,
                failure_kind: Some(failure_kind),
                surface_state,
                unsupported_draw_items: self.unsupported_draw_items,
                stale_drop_count: 0,
                message,
            }),
            error: Some(error),
        }
    }

    #[cfg(test)]
    pub(super) fn new_for_test(window: AnyWindowHandle, generation: u64) -> Self {
        Self {
            window,
            generation,
            unsupported_draw_items: 0,
            started: Instant::now(),
            phase: FramePhase::Begun,
        }
    }
}

pub(super) fn has_supported_glyph_text(prepared: &PreparedFrame) -> bool {
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            kind => {
                if clip_stack.active_clip() != super::clip::ActiveClip::Empty
                    && kind.supported_windows_glyph_text()
                {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn has_supported_box_shapes(prepared: &PreparedFrame) -> bool {
    box_shape_count(prepared) > 0
}

struct ReceiptParts {
    window: AnyWindowHandle,
    surface_generation: u64,
    status: BackendFrameStatus,
    failure_kind: Option<ErrorKind>,
    surface_state: BackendSurfaceState,
    unsupported_draw_items: usize,
    stale_drop_count: u64,
    message: &'static str,
}

pub(super) fn receipt(
    window: AnyWindowHandle,
    surface_generation: u64,
    status: BackendFrameStatus,
    surface_state: BackendSurfaceState,
    unsupported_draw_items: usize,
    stale_drop_count: u64,
    message: &'static str,
) -> BackendFrameReceipt {
    build_receipt(ReceiptParts {
        window,
        surface_generation,
        status,
        failure_kind: None,
        surface_state,
        unsupported_draw_items,
        stale_drop_count,
        message,
    })
}

fn build_receipt(parts: ReceiptParts) -> BackendFrameReceipt {
    BackendFrameReceipt {
        backend: "windows.d3d11.dxgi",
        window: parts.window,
        surface_generation: parts.surface_generation,
        status: parts.status,
        failure_kind: parts.failure_kind,
        surface_state: parts.surface_state,
        unsupported_draw_items: parts.unsupported_draw_items,
        stale_drop_count: parts.stale_drop_count,
        message: parts.message,
    }
}

#[cfg(test)]
pub(crate) fn count_unsupported_draw_items_for_backend(prepared: &PreparedFrame) -> usize {
    unsupported_draw_items(prepared)
}
