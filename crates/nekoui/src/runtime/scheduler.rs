use std::collections::BTreeMap;

use crate::diagnostic::{DirtyLane, DirtyLanes};
use crate::window::WindowId;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WindowSchedulerState {
    dirty_lanes: DirtyLanes,
    pending_redraw: bool,
}

impl WindowSchedulerState {
    pub fn dirty_lanes(&self) -> DirtyLanes {
        self.dirty_lanes
    }

    #[cfg(test)]
    pub fn pending_redraw(&self) -> bool {
        self.pending_redraw
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RedrawRequestOutcome {
    Requested,
    Coalesced,
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
        if state.pending_redraw {
            RedrawRequestOutcome::Coalesced
        } else {
            state.pending_redraw = true;
            RedrawRequestOutcome::Requested
        }
    }

    #[cfg(test)]
    pub fn window_state(&self, window: WindowId) -> Option<&WindowSchedulerState> {
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

        let first = scheduler.request_redraw(window);
        let second = scheduler.request_redraw(window);

        assert_eq!(first, super::RedrawRequestOutcome::Requested);
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
