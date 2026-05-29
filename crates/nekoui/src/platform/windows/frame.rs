use std::time::{Duration, Instant};

use crate::error::{ErrorKind, NekoError, NekoResult};
use crate::platform::{
    BackendFrameReceipt, BackendFrameStatus, BackendSurfaceState, Renderability,
};
use crate::render::PreparedFrame;
use crate::text::FontManager;
use crate::window::AnyWindowHandle;

use super::device::D3d11DeviceState;
use super::glyph::{GlyphAtlas, GlyphUnsupportedReport, prepare_glyph_atlas};
use super::glyph_pipeline::D3d11GlyphMonoPipeline;
use super::pipeline::D3d11SolidRectPipeline;
use super::solid_rect::{collect_solid_rects, solid_rect_count, unsupported_draw_items};
use super::surface::DxgiSurface;

pub(super) struct FrameRecordResources<'a> {
    pub(super) font_manager: &'a FontManager,
    pub(super) glyph_atlas: Option<&'a mut GlyphAtlas>,
    pub(super) glyph_pipeline: Option<&'a mut D3d11GlyphMonoPipeline>,
    pub(super) solid_rect_pipeline: Option<&'a mut D3d11SolidRectPipeline>,
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
        let glyph_atlas = resources.glyph_atlas;
        let mut glyph_pipeline = resources.glyph_pipeline;
        let glyph_draw_count = if has_supported_glyph_text(prepared) {
            let atlas = glyph_atlas.ok_or_else(|| {
                NekoError::unsupported("D3D11 glyph atlas is unavailable for text draw items")
            })?;
            glyph_unsupported.add(prepare_glyph_atlas(
                prepared,
                atlas,
                resources.font_manager,
            )?);
            let pipeline = glyph_pipeline.as_deref_mut().ok_or_else(|| {
                NekoError::unsupported(
                    "D3D11 glyph pipeline is unavailable without checked shader artifacts",
                )
            })?;
            glyph_unsupported.add(pipeline.collect_glyph_draws(prepared, atlas)?);
            let glyph_draw_count = pipeline.glyph_draw_count();
            if glyph_draw_count > 0 {
                pipeline.upload_if_dirty(device, atlas)?;
            }
            glyph_draw_count
        } else {
            0
        };
        self.unsupported_draw_items += glyph_unsupported.skipped_glyph_instances();
        let rects = collect_solid_rects(prepared);
        let mut rect_index = 0;
        let mut glyph_index = 0;
        let mut solid_rect_pipeline = resources.solid_rect_pipeline;
        for item in prepared.draw_items() {
            if matches!(item.kind(), crate::render::DrawItemKind::Rect { color } if color.srgb_channels().is_some())
            {
                let pipeline = solid_rect_pipeline.as_deref_mut().ok_or_else(|| {
                    NekoError::unsupported(
                        "D3D11 solid rect pipeline is unavailable without checked shader artifacts",
                    )
                })?;
                let end = rect_index + 1;
                pipeline.draw(
                    device,
                    surface.render_target_view()?,
                    prepared.context(),
                    &rects[rect_index..end],
                )?;
                rect_index = end;
            } else if item.kind().supported_windows_glyph_text() {
                let pipeline = glyph_pipeline.as_deref_mut().ok_or_else(|| {
                    NekoError::unsupported(
                        "D3D11 glyph pipeline is unavailable without checked shader artifacts",
                    )
                })?;
                let start = glyph_index;
                while glyph_index < glyph_draw_count
                    && pipeline.glyph_draw_order(glyph_index) == item.order()
                {
                    glyph_index += 1;
                }
                if glyph_index > start {
                    pipeline.draw_collected_range(
                        device,
                        surface.render_target_view()?,
                        prepared.context(),
                        start..glyph_index,
                    )?;
                }
            }
        }
        self.phase = FramePhase::Recorded;
        Ok(FrameRecordReport { glyph_unsupported })
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
    prepared
        .draw_items()
        .iter()
        .any(|item| item.kind().supported_windows_glyph_text())
}

pub(super) fn has_supported_solid_rects(prepared: &PreparedFrame) -> bool {
    solid_rect_count(prepared) > 0
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
