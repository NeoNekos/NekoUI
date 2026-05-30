use std::collections::BTreeMap;

use crate::diagnostic::{DirtyLane, DirtyLanes};
use crate::platform::Renderability;
use crate::window::WindowId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowSchedulerState {
    dirty_lanes: DirtyLanes,
    pending_redraw: bool,
    renderability: Renderability,
    suspended_dirty_lanes: DirtyLanes,
    suppressed_redraw_count: u64,
}

impl Default for WindowSchedulerState {
    fn default() -> Self {
        Self {
            dirty_lanes: DirtyLanes::empty(),
            pending_redraw: false,
            renderability: Renderability::SurfaceAbsent,
            suspended_dirty_lanes: DirtyLanes::empty(),
            suppressed_redraw_count: 0,
        }
    }
}

impl WindowSchedulerState {
    pub fn dirty_lanes(&self) -> DirtyLanes {
        self.dirty_lanes
    }

    #[cfg(test)]
    pub fn renderability(&self) -> Renderability {
        self.renderability
    }

    #[cfg(test)]
    pub fn suspended_dirty_lanes(&self) -> DirtyLanes {
        self.suspended_dirty_lanes
    }

    #[cfg(test)]
    pub fn suppressed_redraw_count(&self) -> u64 {
        self.suppressed_redraw_count
    }

    pub(crate) fn pending_redraw(&self) -> bool {
        self.pending_redraw
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RedrawRequestOutcome {
    Requested,
    Coalesced,
    SuppressedNotRenderable,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Scheduler {
    windows: BTreeMap<WindowId, WindowSchedulerState>,
}

impl Scheduler {
    pub fn ensure_window(&mut self, window: WindowId) {
        self.windows.entry(window).or_default();
    }

    pub fn mark_dirty(&mut self, window: WindowId, lane: DirtyLane) {
        self.ensure_window(window);
        if let Some(state) = self.windows.get_mut(&window) {
            state.dirty_lanes.insert(lane.flag());
            if !state.renderability.is_renderable() {
                state.suspended_dirty_lanes.insert(lane.flag());
            }
        }
    }

    pub fn take_dirty_lanes(&mut self, window: WindowId, lanes: DirtyLanes) -> DirtyLanes {
        self.ensure_window(window);
        let Some(state) = self.windows.get_mut(&window) else {
            return DirtyLanes::empty();
        };
        let taken = state.dirty_lanes & lanes;
        state.dirty_lanes.remove(taken);
        taken
    }

    pub(crate) fn request_redraw(&mut self, window: WindowId) -> RedrawRequestOutcome {
        self.ensure_window(window);
        let state = self
            .windows
            .get_mut(&window)
            .expect("window state should exist after ensure_window");
        if !state.renderability.is_renderable() {
            state.suppressed_redraw_count = state.suppressed_redraw_count.saturating_add(1);
            state.suspended_dirty_lanes.insert(state.dirty_lanes);
            RedrawRequestOutcome::SuppressedNotRenderable
        } else if state.pending_redraw {
            RedrawRequestOutcome::Coalesced
        } else {
            state.pending_redraw = true;
            RedrawRequestOutcome::Requested
        }
    }

    pub(crate) fn consume_pending_redraw(&mut self, window: WindowId) -> bool {
        self.ensure_window(window);
        let state = self
            .windows
            .get_mut(&window)
            .expect("window state should exist after ensure_window");
        let was_pending = state.pending_redraw;
        state.pending_redraw = false;
        was_pending
    }

    pub(crate) fn set_renderability(
        &mut self,
        window: WindowId,
        renderability: Renderability,
    ) -> RedrawRequestOutcome {
        self.ensure_window(window);
        let state = self
            .windows
            .get_mut(&window)
            .expect("window state should exist after ensure_window");
        let was_renderable = state.renderability.is_renderable();
        state.renderability = renderability;
        if !renderability.is_renderable() {
            state.pending_redraw = false;
            state.suspended_dirty_lanes.insert(state.dirty_lanes);
            return RedrawRequestOutcome::SuppressedNotRenderable;
        }
        if was_renderable || state.suspended_dirty_lanes.is_empty() {
            return RedrawRequestOutcome::Coalesced;
        }
        state.dirty_lanes.insert(state.suspended_dirty_lanes);
        state.suspended_dirty_lanes = DirtyLanes::empty();
        if state.pending_redraw {
            RedrawRequestOutcome::Coalesced
        } else {
            state.pending_redraw = true;
            RedrawRequestOutcome::Requested
        }
    }

    pub(crate) fn take_platform_redraw_requests(&mut self) -> Vec<WindowId> {
        self.windows
            .iter()
            .filter_map(|(window, state)| {
                (state.pending_redraw && state.renderability.is_renderable()).then_some(*window)
            })
            .collect()
    }

    pub(crate) fn window_state(&self, window: WindowId) -> Option<&WindowSchedulerState> {
        self.windows.get(&window)
    }

    pub fn window_states(&self) -> &BTreeMap<WindowId, WindowSchedulerState> {
        &self.windows
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::DirtyLane;
    use crate::window::WindowId;

    use super::Scheduler;

    #[test]
    fn dirty_lanes_are_explicit_bits() {
        let mut scheduler = Scheduler::default();
        let window = WindowId::new(1);

        scheduler.mark_dirty(window, DirtyLane::Layout);
        scheduler.mark_dirty(window, DirtyLane::Semantics);

        let state = scheduler.window_state(window).unwrap();
        assert!(state.dirty_lanes().contains(DirtyLane::Layout.flag()));
        assert!(state.dirty_lanes().contains(DirtyLane::Semantics.flag()));
    }

    #[test]
    fn redraw_requests_are_coalesced_per_window() {
        let mut scheduler = Scheduler::default();
        let window = WindowId::new(1);

        scheduler.set_renderability(window, crate::platform::Renderability::Renderable);

        let first = scheduler.request_redraw(window);
        let second = scheduler.request_redraw(window);

        assert_eq!(first, super::RedrawRequestOutcome::Requested);
        assert_eq!(second, super::RedrawRequestOutcome::Coalesced);
    }

    #[test]
    fn not_renderable_preserves_dirty_without_pending_redraw_spin() {
        let mut scheduler = Scheduler::default();
        let window = WindowId::new(1);

        scheduler.set_renderability(window, crate::platform::Renderability::ZeroSize);
        scheduler.mark_dirty(window, DirtyLane::Paint);
        let outcome = scheduler.request_redraw(window);
        let state = scheduler.window_state(window).unwrap();

        assert_eq!(
            outcome,
            super::RedrawRequestOutcome::SuppressedNotRenderable
        );
        assert!(!state.pending_redraw());
        assert!(
            state
                .suspended_dirty_lanes()
                .contains(DirtyLane::Paint.flag())
        );
        assert_eq!(state.suppressed_redraw_count(), 1);
    }

    #[test]
    fn restore_after_suspended_dirty_requests_one_redraw() {
        let mut scheduler = Scheduler::default();
        let window = WindowId::new(1);

        scheduler.set_renderability(window, crate::platform::Renderability::ZeroSize);
        scheduler.mark_dirty(window, DirtyLane::Paint);
        scheduler.request_redraw(window);
        let outcome =
            scheduler.set_renderability(window, crate::platform::Renderability::Renderable);
        let second = scheduler.request_redraw(window);

        assert_eq!(outcome, super::RedrawRequestOutcome::Requested);
        assert_eq!(second, super::RedrawRequestOutcome::Coalesced);
    }

    #[test]
    fn dirty_lanes_can_be_taken_and_cleared_by_lane() {
        let mut scheduler = Scheduler::default();
        let window = WindowId::new(1);

        scheduler.mark_dirty(window, DirtyLane::Layout);
        scheduler.mark_dirty(window, DirtyLane::Paint);

        let taken = scheduler.take_dirty_lanes(window, DirtyLane::Layout.flag());
        let state = scheduler.window_state(window).unwrap();

        assert!(taken.contains(DirtyLane::Layout.flag()));
        assert!(!state.dirty_lanes().contains(DirtyLane::Layout.flag()));
        assert!(state.dirty_lanes().contains(DirtyLane::Paint.flag()));
    }
}
