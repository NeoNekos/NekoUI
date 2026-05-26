use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;
use std::time::Duration;

use crate::app::{AppContext, Context, Entity, Render, Subscription};
use crate::diagnostic::signal::SignalId;
use crate::diagnostic::{
    CommandIngressReport, DiagnosticArea, DiagnosticRecord, DiagnosticSeverity, Diagnostics,
    DirtyLane, DirtyLaneReport, DirtyLanes, LayoutPassReport, LayoutPerformanceReport,
    PerformanceReport, RetainedPerformanceReport, ScenePerformanceReport, StylePerformanceReport,
    TextPerformanceReport,
};
use crate::element::{Element, IntoElement};
use crate::error::{ErrorKind, NekoError, NekoResult};
use crate::interaction::{
    ClickEvent, InteractionTarget, PointerEvent, PointerInput, PointerInputKind,
};
use crate::layout::{LayoutPassStats, LayoutSize, compute_layout};
use crate::retained::{DirtyCause, RetainedNodeSnapshot, RetainedTreeDiff, RetainedTreeSnapshot};
use crate::runtime::command::{CommandId, RuntimeCommand, SequencedCommand, WindowCommand};
use crate::runtime::entity_store::EntityKey;
use crate::runtime::scheduler::RedrawRequestOutcome;
use crate::runtime::state::RuntimeState;
use crate::runtime::subscription_store::{SubscriptionCallback, SubscriptionKey};
use crate::scene::{
    SceneCompileInput, compile_scene, scene_generation_for_inputs, scene_publish_is_current,
};
use crate::window::{AnyWindowHandle, WindowHandle, WindowOptions};

#[derive(Default)]
pub(crate) struct Runtime {
    next_sequence: u64,
    queue: VecDeque<SequencedCommand>,
    state: RuntimeState,
    root_views: BTreeMap<crate::window::WindowId, Box<dyn RootViewSlot>>,
    pending_root_rebuilds: BTreeSet<crate::window::WindowId>,
    diagnostics: Diagnostics,
    retained_diff_duration: Duration,
    style_resolve_duration: Duration,
    layout_duration: Duration,
    scene_compile_duration: Duration,
    transaction_depth: usize,
    drain_depth: usize,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Runtime")
            .field("next_sequence", &self.next_sequence)
            .field("queue", &self.queue)
            .field("state", &self.state)
            .field("root_view_count", &self.root_views.len())
            .field("pending_root_rebuilds", &self.pending_root_rebuilds)
            .field("diagnostics", &self.diagnostics)
            .field("retained_diff_duration", &self.retained_diff_duration)
            .field("style_resolve_duration", &self.style_resolve_duration)
            .field("layout_duration", &self.layout_duration)
            .field("scene_compile_duration", &self.scene_compile_duration)
            .field("transaction_depth", &self.transaction_depth)
            .field("drain_depth", &self.drain_depth)
            .finish()
    }
}

trait RootViewSlot {
    fn as_any(&self) -> &dyn Any;
    fn render(&self, runtime: &mut Runtime) -> NekoResult<Element>;
}

struct TypedRootView<T: Render> {
    entity: Entity<T>,
}

impl<T: Render> RootViewSlot for TypedRootView<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&self, runtime: &mut Runtime) -> NekoResult<Element> {
        let value = runtime
            .state
            .entity_store_mut()
            .take_any(self.entity.key())?;
        let mut value = match value.downcast::<T>() {
            Ok(value) => value,
            Err(value) => {
                runtime
                    .state
                    .entity_store_mut()
                    .restore_any(self.entity.key(), value)?;
                return Err(NekoError::invalid_input(
                    "root view type does not match mounted entity",
                ));
            }
        };

        let element = {
            let mut cx = Context::new(runtime, self.entity.key());
            value.render(&mut cx).into_element()
        };
        let validation = element.validate_inputs();
        runtime
            .state
            .entity_store_mut()
            .restore_any(self.entity.key(), value)?;
        validation?;
        Ok(element)
    }
}

fn duration_micros_signal(duration: Duration) -> u64 {
    let micros = duration.as_micros() as u64;
    if micros == 0 && !duration.is_zero() {
        1
    } else {
        micros
    }
}

impl Runtime {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn state(&self) -> &RuntimeState {
        &self.state
    }

    #[cfg(test)]
    pub(crate) fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    pub(crate) fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    pub(crate) fn enqueue(&mut self, command: RuntimeCommand) -> CommandId {
        self.next_sequence += 1;
        let id = CommandId::new(self.next_sequence);
        self.queue.push_back(SequencedCommand::new(id, command));
        self.diagnostics
            .increment_signal(SignalId::RuntimeCommandQueued);
        self.diagnostics
            .add_signal(SignalId::RuntimeQueueDepthTotal, self.queue.len() as u64);
        id
    }

    pub(crate) fn request_notify(&mut self) -> CommandId {
        self.enqueue(RuntimeCommand::Notify)
    }

    pub(crate) fn reserve_entity_key(&mut self) -> EntityKey {
        self.state.entity_store_mut().reserve()
    }

    pub(crate) fn insert_reserved_entity<T: 'static>(
        &mut self,
        key: EntityKey,
        value: T,
    ) -> Entity<T> {
        let owner = Rc::new(());
        self.state
            .entity_store_mut()
            .insert_reserved(key, value, Rc::downgrade(&owner));
        self.diagnostics
            .increment_signal(SignalId::ApiEntityCreated);
        Entity::new(key, owner)
    }

    pub(crate) fn read_entity<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        read: impl FnOnce(&T) -> R,
    ) -> NekoResult<R> {
        self.diagnostics.increment_signal(SignalId::ApiEntityRead);
        match self.state.entity_store().read(entity.key(), read) {
            Ok(value) => Ok(value),
            Err(error) => {
                if error.kind() == ErrorKind::Stale {
                    self.record_api_stale();
                }
                Err(error)
            }
        }
    }

    pub(crate) fn update_entity<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R> {
        self.diagnostics.increment_signal(SignalId::ApiEntityUpdate);
        let value = match self.state.entity_store_mut().take_any(entity.key()) {
            Ok(value) => value,
            Err(error) => {
                if error.kind() == ErrorKind::Stale {
                    self.diagnostics
                        .increment_signal(SignalId::RuntimeStaleDrop);
                    self.record_api_stale();
                }
                return Err(error);
            }
        };
        let mut value = match value.downcast::<T>() {
            Ok(value) => value,
            Err(value) => {
                self.state
                    .entity_store_mut()
                    .restore_any(entity.key(), value)?;
                return Err(NekoError::invalid_input(
                    "entity type does not match handle",
                ));
            }
        };

        self.transaction_depth += 1;
        let result = {
            let mut cx = Context::new(self, entity.key());
            update(&mut value, &mut cx)
        };

        self.state
            .entity_store_mut()
            .restore_any(entity.key(), value)?;
        self.transaction_depth -= 1;
        if self.transaction_depth == 0 && self.drain_depth == 0 {
            self.drain_all()?;
        }
        result
    }

    pub(crate) fn notify_entity(&mut self, source: EntityKey) {
        self.diagnostics.increment_signal(SignalId::ApiEntityNotify);
        self.request_notify();
        let (subscriptions, cancelled) = self.state.subscription_store().live_for_source(source);
        if cancelled > 0 {
            self.diagnostics
                .add_signal(SignalId::ApiSubscriptionCancelled, cancelled);
        }
        for subscription in subscriptions {
            self.enqueue(RuntimeCommand::NotifySubscription { subscription });
        }
    }

    pub(crate) fn observe_entity<T: 'static, U: 'static>(
        &mut self,
        target: EntityKey,
        source: &Entity<U>,
        mut callback: impl FnMut(&mut T, Entity<U>, &mut Context<'_, T>) + 'static,
    ) -> Subscription {
        let owner = Rc::new(());
        let source_key = source.key();
        let source_owner = Rc::downgrade(&source.owner);
        let erased: SubscriptionCallback = Box::new(move |value, runtime, target| {
            let value = value
                .downcast_mut::<T>()
                .ok_or_else(|| NekoError::invalid_input("subscription target type mismatch"))?;
            let source_owner = source_owner.upgrade().ok_or_else(|| {
                runtime.record_api_stale();
                NekoError::stale("subscription source entity is stale")
            })?;
            let source = Entity::new(source_key, source_owner);
            let mut cx = Context::new(runtime, target);
            callback(value, source, &mut cx);
            Ok(())
        });
        let key = self.state.subscription_store_mut().insert(
            source.key(),
            target,
            Rc::downgrade(&owner),
            erased,
        );
        Subscription::new(key, owner)
    }

    pub(crate) fn record_api_stale(&mut self) {
        self.diagnostics.increment_signal(SignalId::ApiHandleStale);
    }

    pub(crate) fn open_window<T: Render>(
        &mut self,
        options: WindowOptions,
        build: impl FnOnce(&mut Context<'_, T>) -> T,
    ) -> NekoResult<WindowHandle<T>> {
        options.viewport_value().validate()?;
        let any = self.state.allocate_window_handle();
        let key = self.reserve_entity_key();
        let mut cx = Context::new(self, key);
        let view = build(&mut cx);
        let entity = self.insert_reserved_entity(key, view);
        self.state.open_window(any, options)?;
        self.root_views
            .insert(any.id(), Box::new(TypedRootView { entity }));
        self.mount_root_view(any, true)?;
        Ok(WindowHandle::new(any))
    }

    pub(crate) fn root_view<T: Render>(
        &mut self,
        handle: WindowHandle<T>,
    ) -> NekoResult<Entity<T>> {
        self.validate_window(handle)?;
        let root_view = self
            .root_views
            .get(&handle.id())
            .ok_or_else(|| NekoError::diagnostic("root view is missing for live window"))?;
        let root_view = root_view
            .as_any()
            .downcast_ref::<TypedRootView<T>>()
            .ok_or_else(|| NekoError::invalid_input("root view type does not match handle"))?;
        Ok(root_view.entity.clone())
    }

    pub(crate) fn request_close_window(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<()> {
        let handle = handle.into();
        self.enqueue(RuntimeCommand::Window(WindowCommand::RequestClose {
            handle,
        }));
        self.drain_if_idle()?;
        Ok(())
    }

    pub(crate) fn close_window(&mut self, handle: impl Into<AnyWindowHandle>) -> NekoResult<()> {
        let handle = handle.into();
        self.enqueue(RuntimeCommand::Window(WindowCommand::Close { handle }));
        self.drain_if_idle()?;
        Ok(())
    }

    pub(crate) fn resize_window(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        logical_size: LayoutSize,
    ) -> NekoResult<()> {
        let handle = handle.into();
        logical_size.validate_viewport()?;
        self.enqueue(RuntimeCommand::Window(WindowCommand::Resize {
            handle,
            logical_size,
        }));
        self.drain_if_idle()?;
        Ok(())
    }

    pub(crate) fn pointer_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: PointerInput,
    ) -> NekoResult<()> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        validate_pointer_input_position(input)?;
        self.enqueue(RuntimeCommand::PointerInput { handle, input });
        self.drain_if_idle()?;
        Ok(())
    }

    pub(crate) fn validate_window(&mut self, handle: impl Into<AnyWindowHandle>) -> NekoResult<()> {
        let handle = handle.into();
        match self.state.validate_window(handle) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.diagnostics
                    .increment_signal(SignalId::RuntimeStaleDrop);
                Err(error)
            }
        }
    }

    pub(crate) fn retained_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<crate::retained::RetainedTreeSnapshot> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        self.state
            .retained_snapshot(handle.id())
            .ok_or_else(|| NekoError::diagnostic("retained tree is missing for live window"))
    }

    pub(crate) fn style_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<crate::style::StyleTreeSnapshot> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        self.state
            .style_snapshot(handle.id())
            .ok_or_else(|| NekoError::diagnostic("style snapshot is missing for live window"))
    }

    pub(crate) fn layout_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<crate::layout::LayoutTreeSnapshot> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        self.state
            .layout_snapshot(handle.id())
            .ok_or_else(|| NekoError::diagnostic("layout snapshot is missing for live window"))
    }

    pub(crate) fn scene_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<crate::scene::PaintScene> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        self.state
            .scene_snapshot(handle.id())
            .ok_or_else(|| NekoError::diagnostic("scene snapshot is missing for live window"))
    }

    pub(crate) fn drain_all(&mut self) -> NekoResult<Vec<CommandId>> {
        self.drain_depth += 1;
        let mut processed = Vec::new();
        while let Some(command) = self.queue.pop_front() {
            let id = command.id();
            let command_result =
                match command.into_inner() {
                    RuntimeCommand::Notify => {
                        let mut notify_count = 1;
                        let mut notify_ids = vec![id];
                        while self.queue.front().is_some_and(|command| {
                            matches!(command.command(), RuntimeCommand::Notify)
                        }) {
                            let notify = self
                                .queue
                                .pop_front()
                                .expect("front command should be present");
                            notify_count += 1;
                            notify_ids.push(notify.id());
                        }
                        self.process_notify(notify_count);
                        for notify_id in notify_ids {
                            self.diagnostics
                                .increment_signal(SignalId::RuntimeCommandProcessed);
                            processed.push(notify_id);
                        }
                        Ok(())
                    }
                    command => self.process(command).map(|()| {
                        self.diagnostics
                            .increment_signal(SignalId::RuntimeCommandProcessed);
                        processed.push(id);
                    }),
                };
            if let Err(error) = command_result {
                self.drain_depth -= 1;
                return Err(error);
            }
            if self.drain_depth == 1
                && let Err(error) = self.rebuild_dirty_roots()
            {
                self.drain_depth -= 1;
                return Err(error);
            }
        }
        self.drain_depth -= 1;
        if self.drain_depth == 0 {
            self.rebuild_dirty_roots()?;
        }
        Ok(processed)
    }

    fn drain_if_idle(&mut self) -> NekoResult<()> {
        if self.drain_depth == 0 {
            self.drain_all()?;
        }
        Ok(())
    }

    pub(crate) fn performance_report(&self) -> PerformanceReport {
        let snapshot = self.diagnostics.snapshot();
        let dirty_lanes = self
            .state
            .scheduler()
            .window_states()
            .keys()
            .map(|window| DirtyLaneReport::new(*window, self.state.reported_dirty_lanes(*window)))
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
            retained: RetainedPerformanceReport {
                node_count: self.state.retained_node_count(),
                diff_count: snapshot.counter("retained.diff"),
                last_diff: self.state.last_retained_diff().clone(),
            },
            style: StylePerformanceReport {
                resolved_node_count: self.state.retained_node_count(),
                resolve_count: snapshot.counter("style.resolve"),
            },
            layout: LayoutPerformanceReport {
                node_count: self.state.layout_node_count(),
                pass_count: snapshot.counter("layout.pass"),
                text_query_count: snapshot.counter("layout.measure_text"),
                text_cache_hits: snapshot.counter("layout.measure_text.cache_hit"),
                text_cache_misses: snapshot.counter("layout.measure_text.cache_miss"),
                blocked_on_text_count: snapshot.counter("layout.blocked_on_text"),
                deferred_count: snapshot.counter("layout.defer"),
                last_pass: LayoutPassReport {
                    node_count: self.state.last_layout_pass().node_count,
                    changed_geometry_count: self.state.last_layout_pass().changed_geometry_count,
                    text_query_count: self.state.last_layout_pass().text_measure.query_count,
                    text_cache_hits: self.state.last_layout_pass().text_measure.cache_hits,
                    text_cache_misses: self.state.last_layout_pass().text_measure.cache_misses,
                    blocked_on_text_count: self
                        .state
                        .last_layout_pass()
                        .layout_blocked_on_text_count,
                    deferred_count: self.state.last_layout_pass().layout_deferred_count,
                },
            },
            text: TextPerformanceReport {
                measure_count: snapshot.counter("text.measure"),
                cache_hits: snapshot.counter("text.measure.cache_hit"),
                cache_misses: snapshot.counter("text.measure.cache_miss"),
                deferred_count: snapshot.counter("text.measure.deferred"),
                failed_count: snapshot.counter("text.measure.failed"),
                total_duration: self.state.last_layout_pass().text_measure.total_duration,
            },
            scene: ScenePerformanceReport {
                compile_count: snapshot.counter("scene.compile"),
                published_node_count: self.state.scene_node_count(),
                last_compile: self.state.last_scene_compile().clone(),
                node_count: self.state.scene_node_count(),
                fragment_count: self.state.last_scene_compile().fragment_count,
                hit_test_entry_count: self.state.last_scene_compile().hit_test_entry_count,
                damage_region_count: self.state.last_scene_compile().damage_region_count,
                resource_demand_count: self.state.last_scene_compile().resource_demand_count,
                stale_drop_count: snapshot.counter("scene.stale_drop"),
                unsupported_fragment_count: self
                    .state
                    .last_scene_compile()
                    .unsupported_fragment_count,
            },
            dirty_lanes,
            phase_durations: BTreeMap::from([
                ("retained.diff", self.retained_diff_duration),
                ("style.resolve", self.style_resolve_duration),
                ("layout.pass", self.layout_duration),
                ("scene.compile", self.scene_compile_duration),
            ]),
        }
    }

    fn request_redraw(&mut self, window: crate::window::WindowId) {
        match self.state.scheduler_mut().request_redraw(window) {
            RedrawRequestOutcome::Requested => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawRequested),
            RedrawRequestOutcome::Coalesced => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawCoalesced),
        }
    }

    fn process(&mut self, command: RuntimeCommand) -> NekoResult<()> {
        match command {
            RuntimeCommand::Notify => {
                self.process_notify(1);
                Ok(())
            }
            RuntimeCommand::NotifySubscription { subscription } => {
                self.process_subscription(subscription)
            }
            RuntimeCommand::PointerInput { handle, input } => {
                self.process_pointer_input(handle, input)
            }
            RuntimeCommand::Window(window_command) => self.process_window(window_command),
        }
    }

    fn process_subscription(&mut self, subscription: SubscriptionKey) -> NekoResult<()> {
        let (target, mut callback) = match self
            .state
            .subscription_store_mut()
            .take_callback(subscription)
        {
            Ok(callback) => callback,
            Err(error) if error.kind() == ErrorKind::Stale => {
                self.diagnostics
                    .increment_signal(SignalId::RuntimeStaleDrop);
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let mut value = match self.state.entity_store_mut().take_any(target) {
            Ok(value) => value,
            Err(error) if error.kind() == ErrorKind::Stale => {
                self.diagnostics
                    .increment_signal(SignalId::RuntimeStaleDrop);
                let _ = self
                    .state
                    .subscription_store_mut()
                    .restore_callback(subscription, callback);
                return Ok(());
            }
            Err(error) => {
                let _ = self
                    .state
                    .subscription_store_mut()
                    .restore_callback(subscription, callback);
                return Err(error);
            }
        };
        let result = callback(value.as_mut(), self, target);
        self.state.entity_store_mut().restore_any(target, value)?;
        match self
            .state
            .subscription_store_mut()
            .restore_callback(subscription, callback)
        {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::Stale => {
                self.diagnostics
                    .increment_signal(SignalId::ApiSubscriptionCancelled);
            }
            Err(error) => return Err(error),
        }
        if result.is_ok() {
            self.diagnostics
                .increment_signal(SignalId::ApiEntityNotificationFlushed);
        }
        result
    }

    fn process_notify(&mut self, notify_count: u64) {
        self.diagnostics
            .add_signal(SignalId::RuntimeNotifyRequested, notify_count);
        for window in self.state.live_window_ids() {
            self.state
                .scheduler_mut()
                .mark_dirty(window, DirtyLane::Build);
            self.state
                .emit_retained_dirty(None, DirtyCause::AppNotified, build_lanes());
            self.pending_root_rebuilds.insert(window);
            self.request_redraw(window);
        }
    }

    fn rebuild_dirty_roots(&mut self) -> NekoResult<()> {
        let windows = std::mem::take(&mut self.pending_root_rebuilds);
        for window in windows {
            let handle = self.state.window_by_id(window)?.handle();
            self.mount_root_view(handle, false)?;
            self.cleanup_stale_interaction_targets(handle)?;
        }
        Ok(())
    }

    fn process_window(&mut self, command: WindowCommand) -> NekoResult<()> {
        match command {
            WindowCommand::RequestClose { handle } => {
                self.state.request_close_window(handle)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                Ok(())
            }
            WindowCommand::Close { handle } => {
                self.state.close_window(handle)?;
                self.root_views.remove(&handle.id());
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                self.state
                    .emit_retained_dirty(None, DirtyCause::WindowClosed, initial_lanes());
                Ok(())
            }
            WindowCommand::Resize {
                handle,
                logical_size,
            } => {
                self.state.resize_window(handle, logical_size)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Layout);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Surface);
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Paint);
                self.request_redraw(handle.id());
                self.execute_layout_if_dirty(handle)?;
                self.execute_scene_if_dirty(handle)?;
                Ok(())
            }
        }
    }

    fn process_pointer_input(
        &mut self,
        handle: AnyWindowHandle,
        input: PointerInput,
    ) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        self.diagnostics
            .increment_signal(SignalId::InputPointerFact);
        self.record_input_fact(handle, input);
        self.cleanup_stale_interaction_targets(handle)?;

        let target = self.hit_target(handle, input);
        match input.kind() {
            PointerInputKind::Move => {
                let previous_hover = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.hover());
                self.state.interaction_mut(handle.id()).set_hover(target);
                if let Err(error) = self.dispatch_pointer(handle, input, target, "pointer_move") {
                    self.state
                        .interaction_mut(handle.id())
                        .set_hover(previous_hover);
                    self.cleanup_stale_interaction_targets(handle)?;
                    return Err(error);
                }
            }
            PointerInputKind::Down => {
                let previous_pressed = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.pressed());
                self.state.interaction_mut(handle.id()).set_pressed(target);
                if let Err(error) = self.dispatch_pointer(handle, input, target, "pointer_down") {
                    self.state
                        .interaction_mut(handle.id())
                        .set_pressed(previous_pressed);
                    self.cleanup_stale_interaction_targets(handle)?;
                    return Err(error);
                }
            }
            PointerInputKind::Up => {
                let pressed = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.pressed());
                self.state.interaction_mut(handle.id()).set_pressed(None);
                if let Err(error) = self.dispatch_pointer(handle, input, target, "pointer_up") {
                    self.cleanup_stale_interaction_targets(handle)?;
                    return Err(error);
                }
                if pressed.is_some() && pressed == target {
                    self.dispatch_click(handle, input, target)?;
                }
            }
            PointerInputKind::Cancel => {
                self.state.interaction_mut(handle.id()).set_pressed(None);
                self.record_input_dispatch(
                    handle,
                    input,
                    target,
                    "pointer_cancel",
                    "cleared",
                    None,
                );
            }
        }
        self.cleanup_stale_interaction_targets(handle)?;
        Ok(())
    }

    fn hit_target(
        &mut self,
        handle: AnyWindowHandle,
        input: PointerInput,
    ) -> Option<InteractionTarget> {
        let target = self.state.scene_snapshot(handle.id()).and_then(|scene| {
            scene
                .hit_test()
                .hit_test(input.position())
                .map(|entry| InteractionTarget::new(entry.node_id(), entry.node_generation()))
        });
        if target.is_some() {
            self.diagnostics.increment_signal(SignalId::InputHit);
        } else {
            self.diagnostics.increment_signal(SignalId::InputMiss);
        }
        self.record_input_dispatch(
            handle,
            input,
            target,
            "hit_test",
            if target.is_some() { "hit" } else { "miss" },
            None,
        );
        target
    }

    fn dispatch_pointer(
        &mut self,
        handle: AnyWindowHandle,
        input: PointerInput,
        target: Option<InteractionTarget>,
        kind: &'static str,
    ) -> NekoResult<()> {
        let Some(target) = target else {
            self.record_input_dispatch(handle, input, None, kind, "miss", None);
            return Ok(());
        };
        let Some(node) = self.current_target_node(handle, target)? else {
            self.record_stale_input_target(handle, target, kind);
            return Ok(());
        };
        let handler = match input.kind() {
            PointerInputKind::Move => node.handlers().pointer_move(),
            PointerInputKind::Down => node.handlers().pointer_down(),
            PointerInputKind::Up => node.handlers().pointer_up(),
            PointerInputKind::Cancel => None,
        };
        let Some(handler) = handler else {
            self.record_input_dispatch(handle, input, Some(target), kind, "no_handler", None);
            return Ok(());
        };

        let event = PointerEvent::new(input);
        let result = {
            let mut cx = AppContext::from_runtime(self);
            handler(&event, &mut cx)
        };
        let error_kind = result.as_ref().err().map(NekoError::kind);
        self.record_input_dispatch(
            handle,
            input,
            Some(target),
            kind,
            if error_kind.is_some() {
                "handler_error"
            } else {
                "dispatched"
            },
            error_kind,
        );
        result
    }

    fn dispatch_click(
        &mut self,
        handle: AnyWindowHandle,
        input: PointerInput,
        target: Option<InteractionTarget>,
    ) -> NekoResult<()> {
        let Some(target) = target else {
            return Ok(());
        };
        let Some(node) = self.current_target_node(handle, target)? else {
            self.record_stale_input_target(handle, target, "click");
            return Ok(());
        };
        let Some(handler) = node.handlers().click() else {
            self.record_input_dispatch(handle, input, Some(target), "click", "no_handler", None);
            return Ok(());
        };

        self.diagnostics
            .increment_signal(SignalId::InputClickDerived);
        let event = ClickEvent::new(PointerEvent::new(input));
        let result = {
            let mut cx = AppContext::from_runtime(self);
            handler(&event, &mut cx)
        };
        let error_kind = result.as_ref().err().map(NekoError::kind);
        self.record_input_dispatch(
            handle,
            input,
            Some(target),
            "click",
            if error_kind.is_some() {
                "handler_error"
            } else {
                "derived"
            },
            error_kind,
        );
        result
    }

    fn current_target_node(
        &self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
    ) -> NekoResult<Option<RetainedNodeSnapshot>> {
        let retained = self.retained_snapshot(handle)?;
        Ok(find_node_by_target(&retained, target).cloned())
    }

    fn cleanup_stale_interaction_targets(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let retained = self.retained_snapshot(handle)?;
        let hover = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.hover());
        let pressed = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.pressed());
        if let Some(target) = hover
            && find_node_by_target(&retained, target).is_none()
        {
            self.state.interaction_mut(handle.id()).set_hover(None);
            self.record_stale_input_target(handle, target, "hover_cleanup");
        }
        if let Some(target) = pressed
            && find_node_by_target(&retained, target).is_none()
        {
            self.state.interaction_mut(handle.id()).set_pressed(None);
            self.record_stale_input_target(handle, target, "pressed_cleanup");
        }
        Ok(())
    }

    fn mount_root_view(&mut self, handle: AnyWindowHandle, initial: bool) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let Some(root_view) = self.root_views.remove(&handle.id()) else {
            return Err(NekoError::diagnostic(
                "root view is missing for live window",
            ));
        };
        let root = root_view.render(self);
        self.root_views.insert(handle.id(), root_view);
        let root = root?;
        self.rebuild_window_root(handle, root)?;
        if initial {
            self.state
                .emit_retained_dirty(None, DirtyCause::WindowOpened, initial_lanes());
        }
        Ok(())
    }

    fn rebuild_window_root(&mut self, handle: AnyWindowHandle, root: Element) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let Some(tree) = self.state.retained_tree_mut(handle.id()) else {
            return Err(NekoError::diagnostic(
                "retained tree is missing for live window",
            ));
        };
        let diff = tree.diff_root(root);
        self.apply_retained_diff(handle, diff);
        self.execute_layout_if_dirty(handle)?;
        self.execute_scene_if_dirty(handle)?;
        Ok(())
    }

    fn execute_layout_if_dirty(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let taken = self
            .state
            .scheduler_mut()
            .take_dirty_lanes(handle.id(), DirtyLane::Layout.flag());
        if !taken.contains(DirtyLane::Layout.flag()) {
            return Ok(());
        }
        self.state.record_consumed_dirty_lanes(handle.id(), taken);

        let viewport = self.state.window(handle)?.viewport();
        let previous = self.state.layout_snapshot(handle.id());
        let Some(tree) = self.state.retained_tree(handle.id()) else {
            return Err(NekoError::diagnostic(
                "retained tree is missing for layout pass",
            ));
        };
        match compute_layout(
            tree.layout_input(),
            viewport,
            previous.as_ref(),
            self.state.font_manager(),
        ) {
            Ok(output) => {
                let stats = output.stats.clone();
                self.record_layout_pass(handle, viewport, &stats, "completed");
                self.state.set_last_layout_pass(stats);
                self.state.set_layout_snapshot(handle.id(), output.snapshot);
                Ok(())
            }
            Err(failure) => {
                let stats = failure.stats().clone();
                self.record_layout_pass(handle, viewport, &stats, "deferred");
                self.diagnostics.record(
                    DiagnosticRecord::new(
                        DiagnosticArea::Layout,
                        DiagnosticSeverity::Warning,
                        ErrorKind::Diagnostic,
                        "layout.defer",
                        "layout pass deferred",
                    )
                    .with_field("window", handle.id().raw().to_string())
                    .with_field("reason", failure.error().message().to_owned())
                    .with_field(
                        "blocked_on_text_count",
                        stats.layout_blocked_on_text_count.to_string(),
                    ),
                );
                self.state.set_last_layout_pass(stats);
                Err(failure.into_error())
            }
        }
    }

    fn execute_scene_if_dirty(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let taken = self
            .state
            .scheduler_mut()
            .take_dirty_lanes(handle.id(), DirtyLane::Paint.flag());
        if !taken.contains(DirtyLane::Paint.flag()) {
            return Ok(());
        }
        self.state.record_consumed_dirty_lanes(handle.id(), taken);

        let retained = self.retained_snapshot(handle)?;
        let style = self.style_snapshot(handle)?;
        let layout = self.layout_snapshot(handle)?;
        let expected_generation = scene_generation_for_inputs(&retained, &style, &layout);
        let previous = self.state.scene_snapshot(handle.id());
        let output = compile_scene(SceneCompileInput {
            retained: &retained,
            style: &style,
            layout: &layout,
            previous: previous.as_ref(),
        });
        let current_retained = self.retained_snapshot(handle)?;
        let current_style = self.style_snapshot(handle)?;
        let current_layout = self.layout_snapshot(handle)?;
        if !scene_publish_is_current(
            expected_generation,
            output.scene.generation(),
            &current_retained,
            &current_style,
            &current_layout,
        ) {
            let mut stats = output.stats;
            stats.stale_drop_count = 1;
            self.record_scene_compile(handle, &stats, "stale_drop");
            self.state.set_last_scene_compile(stats);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Paint);
            return Ok(());
        }
        self.record_scene_compile(handle, &output.stats, "published");
        self.state.set_last_scene_compile(output.stats);
        self.state.set_scene_snapshot(handle.id(), output.scene);
        Ok(())
    }

    fn record_scene_compile(
        &mut self,
        handle: AnyWindowHandle,
        stats: &crate::scene::SceneCompileStats,
        result: &'static str,
    ) {
        self.scene_compile_duration += stats.duration;
        self.diagnostics.increment_signal(SignalId::SceneCompile);
        self.diagnostics
            .add_signal(SignalId::SceneFragmentCount, stats.fragment_count as u64);
        self.diagnostics
            .add_signal(SignalId::SceneDamage, stats.damage_region_count as u64);
        self.diagnostics.add_signal(
            SignalId::SceneResourceDemand,
            stats.resource_demand_count as u64,
        );
        self.diagnostics
            .add_signal(SignalId::SceneStaleDrop, stats.stale_drop_count);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Scene,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "scene.compile",
                "scene compile completed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("result", result)
            .with_field("node_count", stats.node_count.to_string())
            .with_field("fragment_count", stats.fragment_count.to_string())
            .with_field(
                "hit_test_entry_count",
                stats.hit_test_entry_count.to_string(),
            )
            .with_field("damage_region_count", stats.damage_region_count.to_string())
            .with_field(
                "resource_demand_count",
                stats.resource_demand_count.to_string(),
            )
            .with_field(
                "unsupported_fragment_count",
                stats.unsupported_fragment_count.to_string(),
            )
            .with_field("duration_micros", stats.duration.as_micros().to_string()),
        );
    }

    fn record_input_fact(&mut self, handle: AnyWindowHandle, input: PointerInput) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "input.pointer_fact",
                "pointer input fact queued through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("kind", pointer_kind_name(input.kind()))
            .with_field("pointer_id", input.pointer_id().to_string())
            .with_field("x", input.position().x().to_string())
            .with_field("y", input.position().y().to_string()),
        );
    }

    fn record_input_dispatch(
        &mut self,
        handle: AnyWindowHandle,
        input: PointerInput,
        target: Option<InteractionTarget>,
        event_kind: &'static str,
        result: &'static str,
        error_kind: Option<ErrorKind>,
    ) {
        self.diagnostics.increment_signal(SignalId::InputDispatch);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                error_kind.unwrap_or(ErrorKind::Diagnostic),
                "input.dispatch",
                "pointer input dispatched",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("raw_kind", pointer_kind_name(input.kind()))
            .with_field("event_kind", event_kind)
            .with_field("result", result)
            .with_field(
                "error_kind",
                error_kind.map_or_else(|| "none".to_owned(), |kind| format!("{kind:?}")),
            )
            .with_field(
                "target_id",
                target.map_or_else(
                    || "none".to_owned(),
                    |target| target.node_id().raw().to_string(),
                ),
            )
            .with_field(
                "target_generation",
                target.map_or_else(
                    || "none".to_owned(),
                    |target| target.node_generation().raw().to_string(),
                ),
            ),
        );
    }

    fn record_stale_input_target(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        state_kind: &'static str,
    ) {
        self.diagnostics
            .increment_signal(SignalId::InputStaleTarget);
        self.diagnostics
            .increment_signal(SignalId::RuntimeStaleDrop);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Warning,
                ErrorKind::Stale,
                "input.stale_target",
                "stale input target dropped",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("state_kind", state_kind)
            .with_field("target_id", target.node_id().raw().to_string())
            .with_field(
                "expected_generation",
                target.node_generation().raw().to_string(),
            ),
        );
    }

    fn record_layout_pass(
        &mut self,
        handle: AnyWindowHandle,
        viewport: crate::layout::Viewport,
        stats: &LayoutPassStats,
        result: &'static str,
    ) {
        self.layout_duration += stats.duration;
        self.diagnostics.increment_signal(SignalId::LayoutPass);
        self.diagnostics
            .add_signal(SignalId::LayoutNodesTotal, stats.node_count as u64);
        self.diagnostics.add_signal(
            SignalId::LayoutGeometryChanged,
            stats.changed_geometry_count as u64,
        );
        self.diagnostics
            .add_signal(SignalId::LayoutMeasureText, stats.text_measure.query_count);
        self.diagnostics.add_signal(
            SignalId::LayoutMeasureTextCacheHit,
            stats.text_measure.cache_hits,
        );
        self.diagnostics.add_signal(
            SignalId::LayoutMeasureTextCacheMiss,
            stats.text_measure.cache_misses,
        );
        self.diagnostics
            .add_signal(SignalId::TextMeasure, stats.text_measure.measured_count);
        self.diagnostics
            .add_signal(SignalId::TextMeasureCacheHit, stats.text_measure.cache_hits);
        self.diagnostics.add_signal(
            SignalId::TextMeasureCacheMiss,
            stats.text_measure.cache_misses,
        );
        self.diagnostics.add_signal(
            SignalId::LayoutBlockedOnText,
            stats.layout_blocked_on_text_count,
        );
        self.diagnostics
            .add_signal(SignalId::LayoutDefer, stats.layout_deferred_count);
        self.diagnostics.add_signal(
            SignalId::TextMeasureDeferred,
            stats.text_measure.deferred_count,
        );
        self.diagnostics
            .add_signal(SignalId::TextMeasureFailed, stats.text_measure.failed_count);
        self.diagnostics.add_signal(
            SignalId::TextMeasureDurationMicros,
            duration_micros_signal(stats.text_measure.total_duration),
        );
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Layout,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "layout.pass",
                "layout pass completed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("result", result)
            .with_field("node_count", stats.node_count.to_string())
            .with_field(
                "changed_geometry_count",
                stats.changed_geometry_count.to_string(),
            )
            .with_field(
                "viewport_width",
                viewport.logical_size().width().to_string(),
            )
            .with_field(
                "viewport_height",
                viewport.logical_size().height().to_string(),
            )
            .with_field(
                "text_query_count",
                stats.text_measure.query_count.to_string(),
            )
            .with_field("text_cache_hits", stats.text_measure.cache_hits.to_string())
            .with_field(
                "text_cache_misses",
                stats.text_measure.cache_misses.to_string(),
            )
            .with_field(
                "layout_deferred_count",
                stats.layout_deferred_count.to_string(),
            )
            .with_field(
                "blocked_on_text_count",
                stats.layout_blocked_on_text_count.to_string(),
            ),
        );
        if stats.text_measure.query_count > 0 {
            self.diagnostics.record(
                DiagnosticRecord::new(
                    DiagnosticArea::Text,
                    DiagnosticSeverity::Info,
                    ErrorKind::Diagnostic,
                    "text.measure",
                    "text measurement completed",
                )
                .with_field("query_count", stats.text_measure.query_count.to_string())
                .with_field(
                    "measured_count",
                    stats.text_measure.measured_count.to_string(),
                )
                .with_field("cache_hits", stats.text_measure.cache_hits.to_string())
                .with_field("cache_misses", stats.text_measure.cache_misses.to_string())
                .with_field(
                    "deferred_count",
                    stats.text_measure.deferred_count.to_string(),
                )
                .with_field("failed_count", stats.text_measure.failed_count.to_string())
                .with_field(
                    "duration_micros",
                    stats.text_measure.total_duration.as_micros().to_string(),
                ),
            );
            for event in &stats.text_measure.events {
                self.diagnostics.record(
                    DiagnosticRecord::new(
                        DiagnosticArea::Text,
                        DiagnosticSeverity::Debug,
                        ErrorKind::Diagnostic,
                        "text.measure.query",
                        "text measurement query completed",
                    )
                    .with_field("node_id", event.node_id.raw().to_string())
                    .with_field("result", event.result)
                    .with_field("cache", event.cache)
                    .with_field("duration_micros", event.duration.as_micros().to_string())
                    .with_field(
                        "line_count",
                        event
                            .line_count
                            .map_or_else(|| "0".to_owned(), |count| count.to_string()),
                    )
                    .with_field(
                        "min_content_width_bits",
                        event
                            .min_content_width_bits
                            .map_or_else(|| "0".to_owned(), |bits| bits.to_string()),
                    )
                    .with_field(
                        "max_content_width_bits",
                        event
                            .max_content_width_bits
                            .map_or_else(|| "0".to_owned(), |bits| bits.to_string()),
                    ),
                );
            }
            for blocker in &stats.text_measure.blockers {
                self.diagnostics.record(
                    DiagnosticRecord::new(
                        DiagnosticArea::Layout,
                        DiagnosticSeverity::Warning,
                        ErrorKind::Diagnostic,
                        "layout.blocked_on_text",
                        "layout blocked on text measurement",
                    )
                    .with_field("node_id", blocker.node_id.raw().to_string())
                    .with_field("result", blocker.result)
                    .with_field("reason", blocker.reason.to_string())
                    .with_field("dependency", blocker.kind)
                    .with_field("duration_micros", blocker.duration.as_micros().to_string()),
                );
            }
        }
    }

    fn apply_retained_diff(&mut self, handle: AnyWindowHandle, diff: RetainedTreeDiff) {
        let mut lanes = DirtyLanes::empty();
        for dirty in &diff.dirty {
            lanes.insert(dirty.lanes);
        }

        for lane in DirtyLane::all().iter().copied() {
            if lanes.contains(lane.flag()) {
                self.state.scheduler_mut().mark_dirty(handle.id(), lane);
            }
        }

        if !lanes.is_empty() {
            self.request_redraw(handle.id());
        }

        self.retained_diff_duration += diff.duration;
        self.style_resolve_duration += diff.style_duration;
        self.record_retained_diff(&diff);
        self.state.set_last_retained_diff(diff.stats);
        self.state.extend_retained_dirty(diff.dirty);
        self.diagnostics.increment_signal(SignalId::StyleResolve);
    }

    fn record_retained_diff(&mut self, diff: &RetainedTreeDiff) {
        self.diagnostics.increment_signal(SignalId::RetainedDiff);
        self.diagnostics.add_signal(
            SignalId::RetainedNodesTotal,
            diff.stats.new_node_count as u64,
        );
        self.diagnostics
            .add_signal(SignalId::RetainedNodesCreated, diff.stats.created as u64);
        self.diagnostics.add_signal(
            SignalId::RetainedNodesPreserved,
            diff.stats.preserved as u64,
        );
        self.diagnostics
            .add_signal(SignalId::RetainedNodesReplaced, diff.stats.replaced as u64);
        self.diagnostics.add_signal(
            SignalId::RetainedNodesDestroyed,
            diff.stats.destroyed as u64,
        );
        self.diagnostics.add_signal(
            SignalId::StyleNodesResolved,
            diff.stats.new_node_count as u64,
        );
        self.diagnostics
            .add_signal(SignalId::RetainedNodeCreated, diff.stats.created as u64);
        self.diagnostics
            .add_signal(SignalId::RetainedNodeDestroyed, diff.stats.destroyed as u64);
        self.diagnostics
            .add_signal(SignalId::RetainedDirty, diff.dirty.len() as u64);
        self.diagnostics.add_signal(
            SignalId::RetainedKindMismatch,
            diff.stats.kind_mismatches as u64,
        );
        self.diagnostics.add_signal(
            SignalId::RetainedDuplicateKey,
            diff.stats.duplicate_keys as u64,
        );
        self.diagnostics.extend_records(diff.records.clone());
    }
}

fn build_lanes() -> DirtyLanes {
    let mut lanes = DirtyLanes::empty();
    lanes.insert(DirtyLane::Build.flag());
    lanes
}

fn initial_lanes() -> DirtyLanes {
    let mut lanes = build_lanes();
    lanes.insert(DirtyLane::Style.flag());
    lanes.insert(DirtyLane::Layout.flag());
    lanes.insert(DirtyLane::Semantics.flag());
    lanes.insert(DirtyLane::Paint.flag());
    lanes
}

fn pointer_kind_name(kind: PointerInputKind) -> &'static str {
    match kind {
        PointerInputKind::Move => "move",
        PointerInputKind::Down => "down",
        PointerInputKind::Up => "up",
        PointerInputKind::Cancel => "cancel",
    }
}

fn validate_pointer_input_position(input: PointerInput) -> NekoResult<()> {
    let position = input.position();
    if !position.x().is_finite() || !position.y().is_finite() {
        return Err(NekoError::invalid_input(
            "pointer input position must use finite coordinates",
        ));
    }
    Ok(())
}

fn find_node_by_target(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    retained
        .root()
        .and_then(|root| find_node_by_target_from(root, target))
}

fn find_node_by_target_from(
    node: &RetainedNodeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    if node.id() == target.node_id()
        && node.generation() == target.node_generation()
        && node.participation().hit_test()
    {
        return Some(node);
    }
    node.children()
        .iter()
        .find_map(|child| find_node_by_target_from(child, target))
}
