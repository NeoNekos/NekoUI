use std::collections::VecDeque;

use crate::diagnostic::{
    CommandIngressReport, Diagnostics, DirtyLane, DirtyLaneReport, PerformanceReport,
};
use crate::error::NekoResult;
use crate::retained::DirtyCause;
use crate::runtime::command::{CommandId, RuntimeCommand, SequencedCommand, WindowCommand};
use crate::runtime::state::RuntimeState;
use crate::window::{AnyWindowHandle, WindowOptions};

#[derive(Debug, Default)]
pub(crate) struct Runtime {
    next_sequence: u64,
    queue: VecDeque<SequencedCommand>,
    state: RuntimeState,
    diagnostics: Diagnostics,
}

impl Runtime {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn state(&self) -> &RuntimeState {
        &self.state
    }

    pub(crate) fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    pub(crate) fn enqueue(&mut self, command: RuntimeCommand) -> CommandId {
        self.next_sequence += 1;
        let id = CommandId::new(self.next_sequence);
        self.queue.push_back(SequencedCommand::new(id, command));
        self.diagnostics.increment("runtime.command_queued");
        self.diagnostics
            .add("runtime.queue_depth_total", self.queue.len() as u64);
        id
    }

    pub(crate) fn request_notify(&mut self) -> CommandId {
        self.enqueue(RuntimeCommand::Notify)
    }

    pub(crate) fn open_window(&mut self, options: WindowOptions) -> NekoResult<AnyWindowHandle> {
        let handle = self.state.allocate_window_handle();
        self.enqueue(RuntimeCommand::Window(WindowCommand::Open {
            handle,
            options,
        }));
        self.drain_all()?;
        Ok(handle)
    }

    pub(crate) fn request_close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.enqueue(RuntimeCommand::Window(WindowCommand::RequestClose {
            handle,
        }));
        self.drain_all()?;
        Ok(())
    }

    pub(crate) fn close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.enqueue(RuntimeCommand::Window(WindowCommand::Close { handle }));
        self.drain_all()?;
        Ok(())
    }

    pub(crate) fn validate_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        match self.state.validate_window(handle) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.diagnostics.increment("runtime.stale_drop");
                Err(error)
            }
        }
    }

    pub(crate) fn drain_all(&mut self) -> NekoResult<Vec<CommandId>> {
        let mut processed = Vec::new();
        while let Some(command) = self.queue.pop_front() {
            let id = command.id();
            self.process(command)?;
            self.diagnostics.increment("runtime.command_processed");
            processed.push(id);
        }
        Ok(processed)
    }

    pub(crate) fn performance_report(&self) -> PerformanceReport {
        let snapshot = self.diagnostics.snapshot();
        let dirty_lanes = self
            .state
            .scheduler()
            .window_states()
            .iter()
            .map(|(window, state)| DirtyLaneReport::new(*window, state.dirty_lanes()))
            .collect();

        PerformanceReport {
            command_ingress: CommandIngressReport {
                commands_enqueued: snapshot.counter("runtime.command_queued"),
                commands_processed: snapshot.counter("runtime.command_processed"),
                queue_depth: self.queue.len(),
            },
            notify_requests: snapshot.counter("runtime.notify_requested"),
            redraw_requests: snapshot.counter("runtime.redraw_requested"),
            coalesced_redraws: snapshot.counter("runtime.redraw_coalesced"),
            stale_handle_errors: snapshot.counter("runtime.stale_drop"),
            windows_alive: self.state.live_window_count(),
            dirty_lanes,
            phase_durations: Default::default(),
        }
    }

    fn process(&mut self, command: SequencedCommand) -> NekoResult<()> {
        match command.into_inner() {
            RuntimeCommand::Notify => self.process_notify(),
            RuntimeCommand::Window(window_command) => self.process_window(window_command),
        }
    }

    fn process_notify(&mut self) -> NekoResult<()> {
        self.diagnostics.increment("runtime.notify_requested");
        for window in self.state.live_window_ids() {
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Build);
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Style);
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Layout);
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Semantics);
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Paint);
            self.state
                .emit_retained_dirty(None, DirtyCause::AppNotified);
            self.state
                .scheduler_mut()
                .request_redraw(window, &mut self.diagnostics);
        }
        Ok(())
    }

    fn process_window(&mut self, command: WindowCommand) -> NekoResult<()> {
        match command {
            WindowCommand::Open { handle, options } => {
                self.state.open_window(handle, options)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Build);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Style);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Layout);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Paint);
                self.state
                    .emit_retained_dirty(None, DirtyCause::WindowOpened);
                self.state
                    .scheduler_mut()
                    .request_redraw(handle.id(), &mut self.diagnostics);
                Ok(())
            }
            WindowCommand::RequestClose { handle } => {
                self.state.request_close_window(handle)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                Ok(())
            }
            WindowCommand::Close { handle } => {
                self.state.close_window(handle)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                self.state
                    .emit_retained_dirty(None, DirtyCause::WindowClosed);
                Ok(())
            }
        }
    }
}
