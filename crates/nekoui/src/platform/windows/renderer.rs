use std::collections::HashMap;

use winit::window::Window;

use crate::diagnostic::signal::SignalId;
use crate::diagnostic::{DiagnosticArea, DiagnosticRecord, DiagnosticSeverity, Diagnostics};
use crate::error::{ErrorKind, NekoError, NekoResult};
use crate::platform::{
    BackendFrameReceipt, BackendFrameStatus, BackendSurfaceState, PhysicalSize, Renderability,
};
use crate::render::{DrawItemKind, PreparedFrame};
use crate::text::FontManager;
use crate::window::{AnyWindowHandle, WindowId};

use super::clip::{ActiveClip, ClipStack};
use super::device::D3d11DeviceState;
use super::frame::{
    ActiveFrame, FrameRecordResources, FrameReport, has_supported_box_shapes,
    has_supported_glyph_text, receipt,
};
use super::glyph::GlyphUnsupportedReport;
use super::glyph::{GLYPH_ATLAS_HEIGHT, GLYPH_ATLAS_WIDTH, GlyphAtlas, GlyphDrawPlan};
use super::glyph_pipeline::{D3d11GlyphColorPipeline, D3d11GlyphMonoPipeline};
use super::pipeline::D3d11BoxShapePipeline;
use super::surface::DxgiSurface;
use super::window::{NativeWindowHandle, hwnd_from_winit};

pub(crate) struct NativeRenderer {
    device: D3d11DeviceState,
    surfaces: HashMap<WindowId, DxgiSurface>,
    active: HashMap<WindowId, ActiveFrame>,
    box_shape_pipeline: Option<D3d11BoxShapePipeline>,
    mono_glyph_pipeline: Option<D3d11GlyphMonoPipeline>,
    color_glyph_pipeline: Option<D3d11GlyphColorPipeline>,
    mono_glyph_atlas: Option<GlyphAtlas>,
    color_glyph_atlas: Option<GlyphAtlas>,
    glyph_draw_plan: GlyphDrawPlan,
}

impl NativeRenderer {
    pub(crate) fn new(diagnostics: &mut Diagnostics) -> NekoResult<Self> {
        let device = D3d11DeviceState::create()?;
        diagnostics.increment_signal(SignalId::GpuBackendSelected);
        diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Gpu,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "gpu.backend.selected",
                "Windows D3D11/DXGI backend selected",
            )
            .with_field("backend", "windows.d3d11.dxgi")
            .with_field("adapter", device.adapter_summary().to_owned()),
        );
        Ok(Self {
            device,
            surfaces: HashMap::new(),
            active: HashMap::new(),
            box_shape_pipeline: None,
            mono_glyph_pipeline: None,
            color_glyph_pipeline: None,
            mono_glyph_atlas: None,
            color_glyph_atlas: None,
            glyph_draw_plan: GlyphDrawPlan::default(),
        })
    }

    pub(crate) fn register_window(
        &mut self,
        window: &Window,
        handle: AnyWindowHandle,
        physical_size: PhysicalSize,
        generation: u64,
        renderability: Renderability,
        diagnostics: &mut Diagnostics,
    ) -> NekoResult<()> {
        if physical_size.is_zero() || !renderability.is_renderable() {
            self.record_surface_state(
                diagnostics,
                handle,
                generation,
                physical_size,
                BackendSurfaceState::Suspended,
                "register_suspended",
            );
            return Ok(());
        }
        let native = hwnd_from_winit(window)?;
        self.create_surface(handle, native, physical_size, generation, diagnostics)?;
        Ok(())
    }

    fn create_surface(
        &mut self,
        handle: AnyWindowHandle,
        native: NativeWindowHandle,
        physical_size: PhysicalSize,
        generation: u64,
        diagnostics: &mut Diagnostics,
    ) -> NekoResult<()> {
        let surface = DxgiSurface::create(&self.device, native, physical_size, generation)?;
        self.record_surface_state(
            diagnostics,
            handle,
            generation,
            physical_size,
            surface.state(),
            "register_ready",
        );
        self.surfaces.insert(handle.id(), surface);
        Ok(())
    }

    pub(crate) fn resize_or_suspend(
        &mut self,
        window: Option<&Window>,
        handle: AnyWindowHandle,
        physical_size: PhysicalSize,
        generation: u64,
        renderability: Renderability,
        diagnostics: &mut Diagnostics,
    ) -> NekoResult<()> {
        let Some(surface) = self.surfaces.get_mut(&handle.id()) else {
            if physical_size.is_zero() || !renderability.is_renderable() {
                self.record_surface_state(
                    diagnostics,
                    handle,
                    generation,
                    physical_size,
                    BackendSurfaceState::Suspended,
                    "surface_absent_suspended",
                );
                return Ok(());
            }
            if let Some(window) = window {
                let native = hwnd_from_winit(window)?;
                return self.create_surface(handle, native, physical_size, generation, diagnostics);
            }
            self.record_surface_state(
                diagnostics,
                handle,
                generation,
                physical_size,
                BackendSurfaceState::Absent,
                "surface_absent",
            );
            return Ok(());
        };
        surface.resize(&self.device, physical_size, generation, renderability)?;
        let state = surface.state();
        self.record_surface_state(
            diagnostics,
            handle,
            generation,
            physical_size,
            state,
            "resize_or_suspend",
        );
        Ok(())
    }

    pub(crate) fn destroy_window(
        &mut self,
        handle: AnyWindowHandle,
        diagnostics: &mut Diagnostics,
    ) {
        if let Some(active) = self.active.remove(&handle.id()) {
            let report = active.abort(BackendSurfaceState::Destroyed, "window destroyed");
            record_frame_report(diagnostics, report);
        }
        if let Some(mut surface) = self.surfaces.remove(&handle.id()) {
            surface.destroy(&self.device);
            self.record_surface_state(
                diagnostics,
                handle,
                surface.generation(),
                surface.physical_size(),
                BackendSurfaceState::Destroyed,
                "destroyed",
            );
        }
    }

    pub(crate) fn render_prepared_frame(
        &mut self,
        handle: AnyWindowHandle,
        prepared: &PreparedFrame,
        font_manager: &FontManager,
        renderability: Renderability,
        diagnostics: &mut Diagnostics,
    ) -> NekoResult<BackendFrameReceipt> {
        if self.active.contains_key(&handle.id()) {
            let surface_state = self
                .surfaces
                .get(&handle.id())
                .map_or(BackendSurfaceState::Absent, DxgiSurface::state);
            let receipt = receipt(
                handle,
                prepared.generation().surface_generation().unwrap_or(0),
                BackendFrameStatus::Failed,
                surface_state,
                0,
                0,
                "surface already has an active frame",
            );
            record_frame_receipt(diagnostics, &receipt, 0);
            return Ok(receipt);
        }
        let Some(surface) = self.surfaces.get(&handle.id()) else {
            let receipt = receipt(
                handle,
                prepared.generation().surface_generation().unwrap_or(0),
                BackendFrameStatus::NotRenderable,
                BackendSurfaceState::Absent,
                0,
                0,
                "surface is absent",
            );
            record_frame_receipt(diagnostics, &receipt, 0);
            return Ok(receipt);
        };
        let active = match ActiveFrame::begin(handle, surface, prepared, renderability) {
            Ok(active) => active,
            Err(receipt) => {
                record_frame_receipt(diagnostics, &receipt, 0);
                return Ok(receipt);
            }
        };
        self.active.insert(handle.id(), active);
        let mut active = self
            .active
            .remove(&handle.id())
            .ok_or_else(|| NekoError::diagnostic("active frame tracking was lost"))?;
        let Some(surface) = self.surfaces.get(&handle.id()) else {
            let error = NekoError::not_renderable("surface disappeared during frame");
            return self.record_failed_active_frame(
                diagnostics,
                active,
                BackendSurfaceState::Absent,
                error,
                "active frame surface disappeared during record",
            );
        };
        if has_supported_box_shapes(prepared) && self.box_shape_pipeline.is_none() {
            match D3d11BoxShapePipeline::new(&self.device) {
                Ok(pipeline) => self.box_shape_pipeline = Some(pipeline),
                Err(error) => {
                    return self.record_failed_active_frame(
                        diagnostics,
                        active,
                        surface.state(),
                        error,
                        "active frame failed during pipeline materialization",
                    );
                }
            }
            record_pipeline_created(diagnostics, handle, prepared);
        }
        if has_supported_glyph_text(prepared) && self.mono_glyph_atlas.is_none() {
            match GlyphAtlas::new(GLYPH_ATLAS_WIDTH, GLYPH_ATLAS_HEIGHT) {
                Ok(atlas) => self.mono_glyph_atlas = Some(atlas),
                Err(error) => {
                    return self.record_failed_active_frame(
                        diagnostics,
                        active,
                        surface.state(),
                        error,
                        "active frame failed during glyph atlas materialization",
                    );
                }
            }
        }
        let record_report = match active.record(
            &self.device,
            surface,
            prepared,
            FrameRecordResources {
                font_manager,
                mono_glyph_atlas: self.mono_glyph_atlas.as_mut(),
                color_glyph_atlas: &mut self.color_glyph_atlas,
                mono_glyph_pipeline: &mut self.mono_glyph_pipeline,
                color_glyph_pipeline: &mut self.color_glyph_pipeline,
                glyph_draw_plan: &mut self.glyph_draw_plan,
                box_shape_pipeline: self.box_shape_pipeline.as_mut(),
            },
        ) {
            Ok(report) => report,
            Err(error) if error.kind() == ErrorKind::Stale => {
                let report = active.abort(surface.state(), "active frame stale during record");
                let receipt = report.receipt.clone();
                record_frame_report(diagnostics, report);
                return Ok(receipt);
            }
            Err(error) => {
                return self.record_failed_active_frame(
                    diagnostics,
                    active,
                    surface.state(),
                    error,
                    "active frame failed during record",
                );
            }
        };
        if record_report.mono_glyph_pipeline_created {
            record_glyph_pipeline_created(diagnostics, handle, prepared, "core.glyph_mono");
        }
        if record_report.color_glyph_pipeline_created {
            record_glyph_pipeline_created(diagnostics, handle, prepared, "core.glyph_color");
        }
        record_unsupported_draw_items(diagnostics, handle, prepared);
        record_glyph_unsupported(
            diagnostics,
            handle,
            prepared,
            record_report.glyph_unsupported,
        );
        let Some(surface) = self.surfaces.get(&handle.id()) else {
            let error = NekoError::not_renderable("surface disappeared before present");
            return self.record_failed_active_frame(
                diagnostics,
                active,
                BackendSurfaceState::Absent,
                error,
                "active frame surface disappeared before present",
            );
        };
        let report = active.present(surface);
        let receipt = report.receipt.clone();
        let error = report.error.clone();
        record_frame_report(diagnostics, report);
        if let Some(error) = error {
            return finish_failed_active_frame(error, receipt);
        }
        Ok(receipt)
    }

    fn record_failed_active_frame(
        &self,
        diagnostics: &mut Diagnostics,
        active: ActiveFrame,
        surface_state: BackendSurfaceState,
        error: NekoError,
        message: &'static str,
    ) -> NekoResult<BackendFrameReceipt> {
        let report = active.fail(surface_state, message, error.clone());
        let receipt = report.receipt.clone();
        record_frame_report(diagnostics, report);
        finish_failed_active_frame(error, receipt)
    }

    fn record_surface_state(
        &self,
        diagnostics: &mut Diagnostics,
        handle: AnyWindowHandle,
        generation: u64,
        physical_size: PhysicalSize,
        state: BackendSurfaceState,
        transition: &'static str,
    ) {
        diagnostics.increment_signal(SignalId::GpuSurfaceState);
        diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Gpu,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "gpu.surface.state",
                "Windows D3D11/DXGI surface state changed",
            )
            .with_field("backend", "windows.d3d11.dxgi")
            .with_field("window", handle.id().raw().to_string())
            .with_field("surface_generation", generation.to_string())
            .with_field("width", physical_size.width().to_string())
            .with_field("height", physical_size.height().to_string())
            .with_field("state", state.name())
            .with_field("transition", transition),
        );
    }
}

fn finish_failed_active_frame(
    error: NekoError,
    receipt: BackendFrameReceipt,
) -> NekoResult<BackendFrameReceipt> {
    match error.kind() {
        ErrorKind::Unsupported | ErrorKind::NotRenderable | ErrorKind::Stale => Ok(receipt),
        ErrorKind::Cancelled
        | ErrorKind::Unavailable
        | ErrorKind::InvalidInput
        | ErrorKind::ResourceFailure
        | ErrorKind::BackendLost
        | ErrorKind::Diagnostic => Err(error),
    }
}

fn record_glyph_unsupported(
    diagnostics: &mut Diagnostics,
    handle: AnyWindowHandle,
    prepared: &PreparedFrame,
    report: GlyphUnsupportedReport,
) {
    if report.is_empty() {
        return;
    }
    diagnostics.record(
        DiagnosticRecord::new(
            DiagnosticArea::Gpu,
            DiagnosticSeverity::Warning,
            ErrorKind::Unsupported,
            "gpu.glyph.unsupported",
            "Windows D3D11/DXGI v0 glyph path skipped unsupported glyph draws",
        )
        .with_field("backend", "windows.d3d11.dxgi")
        .with_field("window", handle.id().raw().to_string())
        .with_field(
            "surface_generation",
            prepared
                .generation()
                .surface_generation()
                .unwrap_or(0)
                .to_string(),
        )
        .with_field(
            "missing_glyph_demands",
            report.missing_glyph_demands.to_string(),
        )
        .with_field(
            "unsupported_content_demands",
            report.unsupported_content_demands.to_string(),
        )
        .with_field("atlas_full_demands", report.atlas_full_demands.to_string())
        .with_field(
            "atlas_oversize_demands",
            report.atlas_oversize_demands.to_string(),
        )
        .with_field(
            "missing_atlas_entries",
            report.missing_atlas_entries.to_string(),
        ),
    );
}

pub(super) fn record_unsupported_draw_items(
    diagnostics: &mut Diagnostics,
    handle: AnyWindowHandle,
    prepared: &PreparedFrame,
) {
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => {
                clip_stack.push(*clip);
                continue;
            }
            DrawItemKind::ClipPop => {
                clip_stack.pop();
                continue;
            }
            _ if clip_stack.active_clip() == ActiveClip::Empty => continue,
            _ => {}
        }
        let capability = match item.kind() {
            DrawItemKind::BoxShape { shape } => {
                match super::box_shape::unsupported_box_shape_capability(*shape) {
                    Some(capability) => capability,
                    None => continue,
                }
            }
            DrawItemKind::Rect { color } if color.to_current_backend_sdr_srgb_rgba().is_some() => {
                continue;
            }
            DrawItemKind::Rect { .. } => "rect.color_space",
            DrawItemKind::Text { .. } if item.kind().supported_windows_glyph_text() => continue,
            DrawItemKind::Text { color, .. }
                if color.to_current_backend_sdr_srgb_rgba().is_none() =>
            {
                "text.color_space"
            }
            DrawItemKind::Text { .. } => "text.glyph_path",
            DrawItemKind::ClipPush { .. } | DrawItemKind::ClipPop => continue,
            DrawItemKind::Unsupported { capability } => capability,
        };
        diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Gpu,
                DiagnosticSeverity::Warning,
                ErrorKind::Unsupported,
                "gpu.unsupported",
                "Windows D3D11/DXGI v0 box-shape backend skipped an unsupported draw item",
            )
            .with_field("backend", "windows.d3d11.dxgi")
            .with_field("window", handle.id().raw().to_string())
            .with_field("draw_order", item.order().raw().to_string())
            .with_field("node_id", item.node_id().to_string())
            .with_field("capability", capability),
        );
    }
}

fn record_pipeline_created(
    diagnostics: &mut Diagnostics,
    handle: AnyWindowHandle,
    prepared: &PreparedFrame,
) {
    diagnostics.record(
        DiagnosticRecord::new(
            DiagnosticArea::Gpu,
            DiagnosticSeverity::Info,
            ErrorKind::Diagnostic,
            "gpu.pipeline",
            "Windows D3D11 box shape pipeline materialized from checked artifacts",
        )
        .with_field("backend", "windows.d3d11.dxgi")
        .with_field("window", handle.id().raw().to_string())
        .with_field(
            "surface_generation",
            prepared
                .generation()
                .surface_generation()
                .unwrap_or(0)
                .to_string(),
        )
        .with_field("shader", "core.box_shape")
        .with_field("target", "d3d11.sm5.dxbc"),
    );
}

fn record_glyph_pipeline_created(
    diagnostics: &mut Diagnostics,
    handle: AnyWindowHandle,
    prepared: &PreparedFrame,
    shader: &'static str,
) {
    diagnostics.record(
        DiagnosticRecord::new(
            DiagnosticArea::Gpu,
            DiagnosticSeverity::Info,
            ErrorKind::Diagnostic,
            "gpu.pipeline",
            "Windows D3D11 glyph pipeline materialized from checked artifacts",
        )
        .with_field("backend", "windows.d3d11.dxgi")
        .with_field("window", handle.id().raw().to_string())
        .with_field(
            "surface_generation",
            prepared
                .generation()
                .surface_generation()
                .unwrap_or(0)
                .to_string(),
        )
        .with_field("shader", shader)
        .with_field("target", "d3d11.sm5.dxbc"),
    );
}

pub(super) fn record_frame_report(diagnostics: &mut Diagnostics, report: FrameReport) {
    let FrameReport {
        receipt,
        duration,
        error,
    } = report;
    record_frame_receipt(diagnostics, &receipt, duration.as_micros() as u64);
    if let Some(error) = error {
        record_frame_error(diagnostics, &receipt, &error);
    }
}

fn record_frame_receipt(
    diagnostics: &mut Diagnostics,
    receipt: &BackendFrameReceipt,
    duration_micros: u64,
) {
    diagnostics.increment_signal(SignalId::GpuFramePhase);
    match receipt.status {
        BackendFrameStatus::Presented => diagnostics.increment_signal(SignalId::GpuFramePresented),
        BackendFrameStatus::NotRenderable => {
            diagnostics.increment_signal(SignalId::GpuFrameNotRenderable)
        }
        BackendFrameStatus::StaleDropped => {
            diagnostics.increment_signal(SignalId::GpuFrameStaleDrop)
        }
        BackendFrameStatus::Aborted | BackendFrameStatus::Failed => {
            diagnostics.increment_signal(SignalId::GpuRecovery)
        }
    }
    diagnostics.add_signal(
        SignalId::GpuUnsupported,
        receipt.unsupported_draw_items as u64,
    );
    diagnostics.add_signal(SignalId::GpuFrameStaleDrop, receipt.stale_drop_count);
    diagnostics.record(
        DiagnosticRecord::new(
            DiagnosticArea::Gpu,
            DiagnosticSeverity::Info,
            receipt.diagnostic_category(),
            "gpu.frame.phase",
            "Windows D3D11/DXGI frame transaction completed",
        )
        .with_field("backend", receipt.backend)
        .with_field("window", receipt.window.id().raw().to_string())
        .with_field("surface_generation", receipt.surface_generation.to_string())
        .with_field("status", receipt.status.name())
        .with_field("surface_state", receipt.surface_state.name())
        .with_field(
            "unsupported_draw_items",
            receipt.unsupported_draw_items.to_string(),
        )
        .with_field("stale_drop_count", receipt.stale_drop_count.to_string())
        .with_field("duration_micros", duration_micros.to_string())
        .with_field("message", receipt.message),
    );
}

fn record_frame_error(
    diagnostics: &mut Diagnostics,
    receipt: &BackendFrameReceipt,
    error: &NekoError,
) {
    diagnostics.record(
        DiagnosticRecord::new(
            DiagnosticArea::Gpu,
            DiagnosticSeverity::Error,
            error.kind(),
            "gpu.frame.error",
            error.message().to_owned(),
        )
        .with_field("backend", receipt.backend)
        .with_field("window", receipt.window.id().raw().to_string())
        .with_field("surface_generation", receipt.surface_generation.to_string())
        .with_field("status", receipt.status.name())
        .with_field("surface_state", receipt.surface_state.name()),
    );
}
