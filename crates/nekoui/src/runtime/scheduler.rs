use std::collections::BTreeMap;

use crate::diagnostic::{Diagnostics, DirtyLane, DirtyLanes};
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

    pub fn request_redraw(&mut self, window: WindowId, diagnostics: &mut Diagnostics) {
        self.ensure_window(window);
        if let Some(state) = self.windows.get_mut(&window) {
            if state.pending_redraw {
                diagnostics.increment("runtime.redraw_coalesced");
            } else {
                state.pending_redraw = true;
                diagnostics.increment("runtime.redraw_requested");
            }
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
    use crate::diagnostic::Diagnostics;
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
        let mut diagnostics = Diagnostics::default();
        let window = WindowId::new(1);

        scheduler.request_redraw(window, &mut diagnostics);
        scheduler.request_redraw(window, &mut diagnostics);

        assert_eq!(
            diagnostics.snapshot().counter("runtime.redraw_requested"),
            1
        );
        assert_eq!(
            diagnostics.snapshot().counter("runtime.redraw_coalesced"),
            1
        );
    }
}
