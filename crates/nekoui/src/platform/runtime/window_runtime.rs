use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::Instant;

use bitflags::bitflags;
use hashbrown::HashSet;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::window::Window as WinitWindow;

#[cfg(target_os = "linux")]
use crate::platform::window::native::linux::LinuxWindowRoute;
use crate::scene::RetainedTree;
use crate::window::{Window, WindowSize};

use super::super::wgpu::WindowRenderState;
use super::input::{FocusManager, InputRouter, TextInputTarget};

pub(crate) struct RuntimeWindow {
    pub(crate) _internal_id: crate::WindowId,
    pub(crate) public_window: Window,
    pub(crate) native_window: Arc<WinitWindow>,
    pub(crate) template_root: crate::AnyElement,
    pub(crate) referenced_views: HashSet<u64>,
    pub(crate) build_scratch: crate::element::SpecArena,
    pub(crate) retained_tree: RetainedTree,
    pub(crate) compiled_scene: crate::scene::CompiledScene,
    pub(crate) render_state: WindowRenderState,
    pub(crate) native_state: NativeWindowState,
    pub(crate) frame_scheduler: FrameSchedulerState,
    pub(crate) generation_state: WindowGenerationState,
    pub(crate) cursor_position: Option<PhysicalPosition<f64>>,
    pub(crate) focus_manager: FocusManager,
    pub(crate) input_router: InputRouter,
    pub(crate) text_input_target: TextInputTarget,
    #[cfg(target_os = "linux")]
    pub(crate) linux_route: LinuxWindowRoute,
    #[cfg(target_os = "macos")]
    pub(crate) last_drag_click: Option<(Instant, PhysicalPosition<f64>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowGenerationState {
    pub(crate) metrics_generation: u64,
    pub(crate) scene_generation: u64,
    pub(crate) presented_generation: u64,
}

impl WindowGenerationState {
    pub(crate) const fn new_synced() -> Self {
        Self {
            metrics_generation: 1,
            scene_generation: 1,
            presented_generation: 0,
        }
    }

    pub(crate) fn note_metrics_change(&mut self) {
        self.metrics_generation = self.metrics_generation.saturating_add(1);
    }

    pub(crate) fn mark_scene_compiled(&mut self) {
        self.scene_generation = self.metrics_generation;
    }

    pub(crate) fn mark_presented(&mut self) {
        self.presented_generation = self.scene_generation;
    }

    pub(crate) fn scene_is_stale(self) -> bool {
        self.scene_generation != self.metrics_generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NativeWindowState {
    pub(crate) metrics: WindowMetrics,
    pub(crate) visible: bool,
    pub(crate) minimized: bool,
    pub(crate) occluded: bool,
    pub(crate) zero_sized: bool,
}

impl NativeWindowState {
    pub(crate) fn new(
        metrics: WindowMetrics,
        visible: bool,
        minimized: bool,
        occluded: bool,
    ) -> Self {
        Self {
            metrics,
            visible,
            minimized,
            occluded,
            zero_sized: metrics.physical_size.width == 0 || metrics.physical_size.height == 0,
        }
    }

    pub(crate) fn is_presentable(self) -> bool {
        self.visible && !self.occluded && !self.minimized && !self.zero_sized
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct NativeWindowUpdate {
    pub(crate) metrics_changed: bool,
    pub(crate) physical_size_changed: bool,
    pub(crate) became_presentable: bool,
    pub(crate) became_unavailable: bool,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct FrameReasonMask: u8 {
        const USER = 0b0000_0001;
        const RUNTIME_DIRTY = 0b0000_0010;
        const METRICS = 0b0000_0100;
        const VISIBILITY = 0b0000_1000;
        const SURFACE = 0b0001_0000;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameSchedulerState {
    pub(crate) pending_reasons: FrameReasonMask,
    pub(crate) redraw_requested: bool,
    pub(crate) requested_token: u64,
    pub(crate) completed_token: u64,
}

impl FrameSchedulerState {
    pub(crate) const fn new() -> Self {
        Self {
            pending_reasons: FrameReasonMask::empty(),
            redraw_requested: false,
            requested_token: 0,
            completed_token: 0,
        }
    }

    pub(crate) fn mark_pending(&mut self, reasons: FrameReasonMask) {
        self.pending_reasons |= reasons;
    }

    pub(crate) fn has_pending(self) -> bool {
        !self.pending_reasons.is_empty()
    }

    pub(crate) fn mark_redraw_requested(&mut self) {
        self.redraw_requested = true;
        self.requested_token = self.requested_token.saturating_add(1);
    }

    pub(crate) fn begin_redraw(&mut self) {
        self.redraw_requested = false;
    }

    pub(crate) fn mark_unavailable(&mut self) {
        self.redraw_requested = false;
    }

    pub(crate) fn mark_presented(&mut self) {
        self.completed_token = self.requested_token;
        self.pending_reasons = FrameReasonMask::empty();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowMetrics {
    pub(crate) logical_size: WindowSize,
    pub(crate) physical_size: WindowSize,
    pub(crate) scale_factor: f64,
}

pub(crate) fn current_window_metrics(window: &WinitWindow) -> WindowMetrics {
    metrics_from_parts(window.inner_size(), window.scale_factor())
}

pub(crate) fn metrics_from_physical_size(
    runtime_window: &RuntimeWindow,
    physical_size: PhysicalSize<u32>,
) -> WindowMetrics {
    metrics_from_parts(physical_size, runtime_window.native_window.scale_factor())
}

pub(crate) fn metrics_from_scale_change(
    runtime_window: &RuntimeWindow,
    scale_factor: f64,
) -> WindowMetrics {
    metrics_from_parts(runtime_window.native_window.inner_size(), scale_factor)
}

pub(crate) fn window_depends_on_dirty_views(
    runtime_window: &RuntimeWindow,
    dirty_views: &hashbrown::HashMap<u64, crate::scene::DirtyLaneMask>,
) -> bool {
    dirty_views
        .keys()
        .any(|view_id| runtime_window.referenced_views.contains(view_id))
}

pub(crate) fn window_metrics_changed(
    runtime_window: &RuntimeWindow,
    metrics: WindowMetrics,
) -> bool {
    runtime_window.public_window.content_size() != metrics.logical_size
        || runtime_window.public_window.physical_size() != metrics.physical_size
        || (runtime_window.public_window.scale_factor() - metrics.scale_factor).abs() > f64::EPSILON
}

pub(crate) fn window_can_present(runtime_window: &RuntimeWindow) -> bool {
    runtime_window.native_state.is_presentable()
}

pub(crate) fn update_native_window_metrics(
    runtime_window: &mut RuntimeWindow,
    metrics: WindowMetrics,
) -> NativeWindowUpdate {
    let (next, update) = transition_native_window_state(
        runtime_window.native_state,
        metrics,
        runtime_window.public_window.visible(),
        runtime_window.native_window.is_minimized().unwrap_or(false),
        runtime_window.native_state.occluded,
    );
    runtime_window.native_state = next;
    update
}

pub(crate) fn update_native_window_occlusion(
    runtime_window: &mut RuntimeWindow,
    occluded: bool,
) -> NativeWindowUpdate {
    let (next, update) = transition_native_window_state(
        runtime_window.native_state,
        runtime_window.native_state.metrics,
        runtime_window.native_state.visible,
        runtime_window.native_state.minimized,
        occluded,
    );
    runtime_window.native_state = next;
    update
}

pub(crate) fn update_native_window_visibility(
    runtime_window: &mut RuntimeWindow,
    visible: bool,
) -> NativeWindowUpdate {
    let (next, update) = transition_native_window_state(
        runtime_window.native_state,
        runtime_window.native_state.metrics,
        visible,
        runtime_window.native_state.minimized,
        runtime_window.native_state.occluded,
    );
    runtime_window.native_state = next;
    update
}

pub(crate) fn refresh_native_window_snapshot(
    runtime_window: &mut RuntimeWindow,
    metrics: WindowMetrics,
) -> NativeWindowUpdate {
    let (next, update) = transition_native_window_state(
        runtime_window.native_state,
        metrics,
        runtime_window.public_window.visible(),
        runtime_window.native_window.is_minimized().unwrap_or(false),
        runtime_window.native_state.occluded,
    );
    runtime_window.native_state = next;
    update
}

fn transition_native_window_state(
    previous: NativeWindowState,
    metrics: WindowMetrics,
    visible: bool,
    minimized: bool,
    occluded: bool,
) -> (NativeWindowState, NativeWindowUpdate) {
    let next = NativeWindowState::new(metrics, visible, minimized, occluded);
    let previous_presentable = previous.is_presentable();
    let next_presentable = next.is_presentable();

    (
        next,
        NativeWindowUpdate {
            metrics_changed: previous.metrics.logical_size != next.metrics.logical_size
                || previous.metrics.physical_size != next.metrics.physical_size
                || (previous.metrics.scale_factor - next.metrics.scale_factor).abs() > f64::EPSILON,
            physical_size_changed: previous.metrics.physical_size != next.metrics.physical_size,
            became_presentable: !previous_presentable && next_presentable,
            became_unavailable: previous_presentable && !next_presentable,
        },
    )
}

fn metrics_from_parts(physical_size: PhysicalSize<u32>, scale_factor: f64) -> WindowMetrics {
    let scale_factor = sanitize_scale_factor(scale_factor);
    let logical_width = (f64::from(physical_size.width) / scale_factor)
        .round()
        .max(1.0) as u32;
    let logical_height = (f64::from(physical_size.height) / scale_factor)
        .round()
        .max(1.0) as u32;

    WindowMetrics {
        logical_size: WindowSize::new(logical_width, logical_height),
        physical_size: WindowSize::new(physical_size.width, physical_size.height),
        scale_factor,
    }
}

fn sanitize_scale_factor(scale_factor: f64) -> f64 {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use crate::window::WindowSize;

    use super::{
        FrameReasonMask, FrameSchedulerState, NativeWindowState, WindowGenerationState,
        WindowMetrics, transition_native_window_state,
    };

    #[test]
    fn metrics_difference_detects_zero_and_scale_changes() {
        let old = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(1600, 1200),
            scale_factor: 2.0,
        };
        let new = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(0, 0),
            scale_factor: 2.0,
        };
        assert_ne!(old.physical_size, new.physical_size);
    }

    #[test]
    fn generation_state_marks_scene_stale_until_compile_catches_up() {
        let mut state = WindowGenerationState::new_synced();
        assert!(!state.scene_is_stale());

        state.note_metrics_change();
        assert!(state.scene_is_stale());

        state.mark_scene_compiled();
        assert!(!state.scene_is_stale());

        state.mark_presented();
        assert_eq!(state.presented_generation, state.scene_generation);
    }

    #[test]
    fn frame_scheduler_tracks_pending_and_completed_tokens() {
        let mut scheduler = FrameSchedulerState::new();
        assert!(!scheduler.has_pending());

        scheduler.mark_pending(FrameReasonMask::METRICS | FrameReasonMask::SURFACE);
        assert!(scheduler.has_pending());

        scheduler.mark_redraw_requested();
        assert!(scheduler.redraw_requested);
        assert_eq!(scheduler.requested_token, 1);

        scheduler.begin_redraw();
        assert!(!scheduler.redraw_requested);

        scheduler.mark_presented();
        assert!(!scheduler.has_pending());
        assert_eq!(scheduler.completed_token, 1);
    }

    #[test]
    fn native_window_state_presentability_tracks_zero_size_and_occlusion() {
        let metrics = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(1600, 1200),
            scale_factor: 2.0,
        };
        let visible = true;

        let stable = super::NativeWindowState::new(metrics, visible, false, false);
        assert!(stable.is_presentable());

        let occluded = super::NativeWindowState::new(metrics, visible, false, true);
        assert!(!occluded.is_presentable());

        let minimized = super::NativeWindowState::new(metrics, visible, true, false);
        assert!(!minimized.is_presentable());

        let zero = super::NativeWindowState::new(
            WindowMetrics {
                logical_size: WindowSize::new(1, 1),
                physical_size: WindowSize::new(0, 0),
                scale_factor: 1.0,
            },
            visible,
            false,
            false,
        );
        assert!(!zero.is_presentable());
    }

    #[test]
    fn native_window_transition_reports_restore_without_metric_change() {
        let metrics = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(1600, 1200),
            scale_factor: 2.0,
        };
        let previous = NativeWindowState::new(metrics, true, true, false);
        let (next, update) = transition_native_window_state(previous, metrics, true, false, false);

        assert!(!update.metrics_changed);
        assert!(!update.physical_size_changed);
        assert!(update.became_presentable);
        assert!(!update.became_unavailable);
        assert!(!next.minimized);
    }

    #[test]
    fn native_window_transition_reports_show_without_metric_change() {
        let metrics = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(1600, 1200),
            scale_factor: 2.0,
        };
        let previous = NativeWindowState::new(metrics, false, false, false);
        let (next, update) = transition_native_window_state(previous, metrics, true, false, false);

        assert!(!update.metrics_changed);
        assert!(update.became_presentable);
        assert!(!update.became_unavailable);
        assert!(next.visible);
    }
}
