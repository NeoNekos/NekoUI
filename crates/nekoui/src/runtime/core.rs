use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;
use std::time::Duration;

use crate::app::{AppContext, Context, Entity, Render, Subscription};
use crate::diagnostic::signal::SignalId;
use crate::diagnostic::{
    CommandIngressReport, DiagnosticArea, DiagnosticRecord, DiagnosticSeverity, Diagnostics,
    DirtyLane, DirtyLaneReport, DirtyLanes, LayoutPassReport, LayoutPerformanceReport,
    PerformanceReport, RenderFrameGraphReport, RenderPerformanceReport, RetainedPerformanceReport,
    ScenePerformanceReport, StylePerformanceReport, TextPerformanceReport,
};
use crate::element::{Element, ElementKind, IntoElement};
use crate::error::{ErrorKind, NekoError, NekoResult};
use crate::interaction::{
    ClickEvent, ImeInput, InteractionTarget, Key, KeyEvent, KeyInput, KeyInputKind, Modifiers,
    PointerEvent, PointerInput, PointerInputKind, ScrollDelta, TextInput, TextInputPurpose,
    TextRange, WheelInput, WindowFocusInput,
};
#[cfg(test)]
use crate::layout::ScrollGeometry;
use crate::layout::{
    LayoutNodeSnapshot, LayoutPassStats, LayoutPoint, LayoutRect, LayoutSize, compute_layout,
    text_viewport_placement,
};
#[cfg(target_os = "windows")]
use crate::platform::NativeRenderer;
use crate::platform::{ImePlatformRequest, PlatformFact, Renderability};
use crate::render::{FrameGraphStats, PreparedFrameContext, prepare_frame_graph_for_surface};
use crate::retained::{DirtyCause, RetainedNodeSnapshot, RetainedTreeDiff, RetainedTreeSnapshot};
use crate::runtime::command::{CommandId, RuntimeCommand, SequencedCommand, WindowCommand};
use crate::runtime::entity_store::EntityKey;
use crate::runtime::scheduler::RedrawRequestOutcome;
use crate::runtime::state::RuntimeState;
use crate::runtime::subscription_store::{SubscriptionCallback, SubscriptionKey};
use crate::scene::{
    HitTestEntry, SceneCompileInput, compile_scene, scene_generation_for_inputs_with_interaction,
    scene_publish_is_current_with_interaction,
};
use crate::semantics::{
    SemanticBuildInput, SemanticBuildStats, build_semantic_snapshot,
    semantic_publish_is_current_with_interaction,
};
use crate::text::{TextEditOutcome, TextRangeError};
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
    semantic_build_duration: Duration,
    scene_compile_duration: Duration,
    render_prepare_duration: Duration,
    transaction_depth: usize,
    drain_depth: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScrollGeometryProbe {
    pub(crate) scroll_target: InteractionTarget,
    pub(crate) observed_target: InteractionTarget,
    pub(crate) scroll: Option<ScrollGeometryProbeScroll>,
    pub(crate) current_offset: LayoutPoint,
    pub(crate) hit_target: Option<InteractionTarget>,
    pub(crate) hit_path: Vec<InteractionTarget>,
    pub(crate) paint_bounds: Vec<LayoutRect>,
    pub(crate) semantic_bounds: Option<LayoutRect>,
    pub(crate) ime_caret_rect: Option<LayoutRect>,
    pub(crate) ime_candidate_rect: Option<LayoutRect>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScrollGeometryProbeScroll {
    pub(crate) viewport: LayoutRect,
    pub(crate) content_extent: LayoutSize,
    pub(crate) max_offset: LayoutPoint,
}

#[cfg(test)]
impl ScrollGeometryProbeScroll {
    fn new(geometry: ScrollGeometry) -> Self {
        Self {
            viewport: geometry.viewport(),
            content_extent: geometry.content_extent(),
            max_offset: geometry.max_offset(),
        }
    }
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
            .field("semantic_build_duration", &self.semantic_build_duration)
            .field("scene_compile_duration", &self.scene_compile_duration)
            .field("render_prepare_duration", &self.render_prepare_duration)
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

    #[cfg(test)]
    pub(crate) fn state(&self) -> &RuntimeState {
        &self.state
    }

    #[cfg(test)]
    pub(crate) fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    #[cfg(test)]
    pub(crate) fn scroll_offset(
        &self,
        handle: impl Into<AnyWindowHandle>,
        target: InteractionTarget,
    ) -> NekoResult<LayoutPoint> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        Ok(self
            .state
            .interaction(handle.id())
            .map_or(LayoutPoint::ZERO, |state| state.scroll_offset(target)))
    }

    pub(crate) fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn diagnostics_mut(&mut self) -> &mut Diagnostics {
        &mut self.diagnostics
    }

    pub(crate) fn enqueue(&mut self, command: RuntimeCommand) -> CommandId {
        self.next_sequence += 1;
        let id = CommandId::new(self.next_sequence);
        if matches!(command, RuntimeCommand::PlatformFact(_)) {
            self.diagnostics
                .increment_signal(SignalId::PlatformFactQueued);
        }
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
        #[cfg(test)]
        self.state
            .set_window_renderability(any, Renderability::Renderable)?;
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

    #[cfg(test)]
    pub(crate) fn key_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: KeyInput,
    ) -> NekoResult<()> {
        let handle = handle.into();
        self.ingest_platform_fact(PlatformFact::KeyInput { handle, input })?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn ime_requests(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<Vec<ImePlatformRequest>> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        Ok(self.state.peek_ime_requests(handle.id()).to_vec())
    }

    #[cfg(test)]
    pub(crate) fn wheel_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: WheelInput,
    ) -> NekoResult<()> {
        let handle = handle.into();
        self.ingest_platform_fact(PlatformFact::WheelInput { handle, input })?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn scroll_geometry_probe(
        &self,
        handle: impl Into<AnyWindowHandle>,
        scroll_target: InteractionTarget,
        observed_target: InteractionTarget,
        hit_position: LayoutPoint,
    ) -> NekoResult<ScrollGeometryProbe> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        let scroll = self
            .layout_scroll_geometry(handle, scroll_target)?
            .map(ScrollGeometryProbeScroll::new);
        let current_offset = self
            .state
            .interaction(handle.id())
            .map_or(LayoutPoint::ZERO, |state| {
                state.scroll_offset(scroll_target)
            });
        let hit_entry = self.hit_entry_at(handle, hit_position);
        let hit_target = hit_entry
            .as_ref()
            .map(|entry| InteractionTarget::new(entry.node_id(), entry.node_generation()));
        let hit_path = hit_entry
            .as_ref()
            .map(|entry| {
                entry
                    .path()
                    .iter()
                    .map(|path| InteractionTarget::new(path.node_id(), path.node_generation()))
                    .collect()
            })
            .unwrap_or_default();
        let paint_bounds = self
            .state
            .scene_snapshot(handle.id())
            .map(|scene| {
                scene
                    .fragments()
                    .iter()
                    .filter(|fragment| {
                        fragment.node_id() == observed_target.node_id()
                            && fragment.node_generation() == observed_target.node_generation()
                    })
                    .map(|fragment| fragment.rect())
                    .collect()
            })
            .unwrap_or_default();
        let semantic_bounds = self
            .state
            .semantic_snapshot(handle.id())
            .and_then(|snapshot| snapshot.bounds_for_retained_target(observed_target));
        let ime_caret_rect = self.text_input_cursor_area(handle, observed_target);
        let ime_candidate_rect = self
            .state
            .peek_ime_requests(handle.id())
            .iter()
            .rev()
            .find_map(|request| match request {
                ImePlatformRequest::CursorArea { rect } => Some(*rect),
                _ => None,
            });

        Ok(ScrollGeometryProbe {
            scroll_target,
            observed_target,
            scroll,
            current_offset,
            hit_target,
            hit_path,
            paint_bounds,
            semantic_bounds,
            ime_caret_rect,
            ime_candidate_rect,
        })
    }

    #[cfg(test)]
    pub(crate) fn window_focus_changed(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: WindowFocusInput,
    ) -> NekoResult<()> {
        let handle = handle.into();
        self.ingest_platform_fact(PlatformFact::WindowFocusChanged { handle, input })?;
        Ok(())
    }

    pub(crate) fn ingest_platform_fact(&mut self, fact: PlatformFact) -> NekoResult<CommandId> {
        self.validate_platform_fact(&fact)?;
        let id = self.enqueue(RuntimeCommand::PlatformFact(fact));
        self.drain_if_idle()?;
        Ok(id)
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

    #[cfg(test)]
    pub(crate) fn semantic_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<crate::semantics::SemanticTreeSnapshot> {
        let handle = handle.into();
        self.state.validate_window(handle)?;
        self.state
            .semantic_snapshot(handle.id())
            .ok_or_else(|| NekoError::diagnostic("semantic snapshot is missing for live window"))
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
            self.execute_semantics_without_redraw_if_ready()?;
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
            render: RenderPerformanceReport {
                frame_graph_count: snapshot.counter("render.frame_graph"),
                pass_count: snapshot.counter("render.pass"),
                upload_plan_count: snapshot.counter("render.upload.plan"),
                layer_count: snapshot.counter("render.layer"),
                stale_drop_count: snapshot.counter("render.stale_drop"),
                unsupported_count: snapshot.counter("render.unsupported"),
                prepared_frame_count: self.state.prepared_frame_count(),
                last_frame_graph: RenderFrameGraphReport {
                    surface_generation: self.state.last_frame_graph().surface_generation,
                    pass_count: self.state.last_frame_graph().pass_count,
                    draw_item_count: self.state.last_frame_graph().draw_item_count,
                    upload_intent_count: self.state.last_frame_graph().upload_intent_count,
                    layer_count: self.state.last_frame_graph().layer_count,
                    box_shape_count: self.state.last_frame_graph().box_shape_count,
                    unsupported_fragment_count: self
                        .state
                        .last_frame_graph()
                        .unsupported_fragment_count,
                    stale_drop_count: self.state.last_frame_graph().stale_drop_count,
                    duration: self.state.last_frame_graph().duration,
                },
            },
            gpu: crate::diagnostic::GpuPerformanceReport {
                backend_selected_count: snapshot.counter("gpu.backend.selected"),
                surface_state_count: snapshot.counter("gpu.surface.state"),
                frame_phase_count: snapshot.counter("gpu.frame.phase"),
                presented_count: snapshot.counter("gpu.frame.presented"),
                not_renderable_count: snapshot.counter("gpu.frame.not_renderable"),
                stale_drop_count: snapshot.counter("gpu.frame.stale_drop"),
                unsupported_count: snapshot.counter("gpu.unsupported"),
                recovery_count: snapshot.counter("gpu.recovery"),
            },
            dirty_lanes,
            phase_durations: BTreeMap::from([
                ("retained.diff", self.retained_diff_duration),
                ("style.resolve", self.style_resolve_duration),
                ("layout.pass", self.layout_duration),
                ("semantics.build", self.semantic_build_duration),
                ("scene.compile", self.scene_compile_duration),
                ("render.frame_graph", self.render_prepare_duration),
            ]),
        }
    }

    pub(crate) fn take_platform_redraw_requests(&mut self) -> Vec<AnyWindowHandle> {
        self.state
            .scheduler_mut()
            .take_platform_redraw_requests()
            .into_iter()
            .filter_map(|window| {
                self.state
                    .window_by_id(window)
                    .ok()
                    .map(|record| record.handle())
            })
            .collect()
    }

    pub(crate) fn take_platform_ime_requests(
        &mut self,
        handle: AnyWindowHandle,
    ) -> NekoResult<Vec<ImePlatformRequest>> {
        self.state.validate_window(handle)?;
        Ok(self.state.take_ime_requests(handle.id()))
    }

    pub(crate) fn take_platform_close_requests(&self) -> Vec<AnyWindowHandle> {
        self.state.closing_windows_for_platform()
    }
    pub(crate) fn live_windows_for_platform(&self) -> Vec<crate::window::WindowRecord> {
        self.state.live_windows()
    }

    pub(crate) fn windows_needing_native_creation(&self) -> Vec<crate::window::WindowRecord> {
        self.state.windows_needing_native_creation()
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn window_record_for_platform(
        &self,
        handle: AnyWindowHandle,
    ) -> NekoResult<crate::window::WindowRecord> {
        self.state.window(handle).cloned()
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn render_prepared_frame_for_platform(
        &mut self,
        renderer: &mut NativeRenderer,
        handle: AnyWindowHandle,
    ) -> NekoResult<Option<crate::platform::BackendFrameReceipt>> {
        let record = self.state.window(handle)?.clone();
        let Some(prepared) = self.state.prepared_frame_snapshot(handle.id()) else {
            return Ok(None);
        };
        let receipt = renderer.render_prepared_frame(
            handle,
            &prepared,
            self.state.font_manager(),
            record.renderability(),
            &mut self.diagnostics,
        )?;
        Ok(Some(receipt))
    }

    fn request_redraw(&mut self, window: crate::window::WindowId) {
        match self.state.scheduler_mut().request_redraw(window) {
            RedrawRequestOutcome::Requested => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawRequested),
            RedrawRequestOutcome::Coalesced => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawCoalesced),
            RedrawRequestOutcome::SuppressedNotRenderable => {
                self.diagnostics
                    .increment_signal(SignalId::RuntimeRedrawSuppressed);
                self.diagnostics
                    .increment_signal(SignalId::RuntimeNotRenderable);
            }
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
            RuntimeCommand::PlatformFact(fact) => self.process_platform_fact(fact),
            RuntimeCommand::Window(window_command) => self.process_window(window_command),
        }
    }

    fn process_platform_fact(&mut self, fact: PlatformFact) -> NekoResult<()> {
        self.diagnostics
            .increment_signal(SignalId::PlatformFactProcessed);
        match fact {
            PlatformFact::WindowCreated { handle } => {
                self.state.mark_native_window_created(handle)?;
                self.diagnostics
                    .increment_signal(SignalId::PlatformWindowCreated);
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                Ok(())
            }
            PlatformFact::WindowShown { handle } => {
                self.state.show_window(handle)?;
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                Ok(())
            }
            PlatformFact::CloseRequested { handle } => {
                self.state.request_close_window(handle)?;
                self.state
                    .scheduler_mut()
                    .mark_dirty(handle.id(), DirtyLane::Semantics);
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                self.enqueue(RuntimeCommand::PlatformFact(PlatformFact::CloseConfirmed {
                    handle,
                }));
                Ok(())
            }
            PlatformFact::CloseConfirmed { handle } => {
                self.state.confirm_close_window(handle)?;
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                Ok(())
            }
            PlatformFact::Destroyed { handle } => {
                self.diagnostics
                    .increment_signal(SignalId::PlatformWindowDestroyed);
                self.destroy_window(handle)
            }
            PlatformFact::LogicalSizeChanged {
                handle,
                logical_size,
            } => {
                logical_size.validate_viewport()?;
                self.apply_window_resize(handle, logical_size, true)
            }
            PlatformFact::PhysicalSizeChanged {
                handle,
                physical_size,
            } => {
                let changed = self.state.set_window_physical_size(handle, physical_size)?;
                let renderability = if physical_size.is_zero() {
                    Renderability::ZeroSize
                } else {
                    Renderability::Renderable
                };
                self.apply_renderability(handle, renderability)?;
                if changed {
                    self.diagnostics
                        .increment_signal(SignalId::SurfaceGenerationBumped);
                    self.mark_size_or_scale_changed(handle);
                }
                Ok(())
            }
            PlatformFact::ScaleFactorChanged {
                handle,
                scale_factor,
            } => {
                let changed = self.state.rescale_window(handle, scale_factor)?;
                if changed {
                    self.diagnostics
                        .increment_signal(SignalId::SurfaceGenerationBumped);
                    self.mark_size_or_scale_changed(handle);
                }
                Ok(())
            }
            PlatformFact::Minimized { handle } => {
                self.state.minimize_window(handle)?;
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                self.diagnostics
                    .increment_signal(SignalId::RuntimeNotRenderable);
                Ok(())
            }
            PlatformFact::Restored { handle } => {
                let requested = self.state.restore_window(handle)?;
                self.diagnostics
                    .increment_signal(SignalId::WindowLifecycleTransition);
                if requested {
                    self.diagnostics
                        .increment_signal(SignalId::RuntimeRedrawRequested);
                }
                Ok(())
            }
            PlatformFact::RenderabilityChanged {
                handle,
                renderability,
            } => self.apply_renderability(handle, renderability),
            PlatformFact::RedrawRequested { handle } => {
                self.state.validate_window(handle)?;
                if self
                    .state
                    .scheduler_mut()
                    .consume_pending_redraw(handle.id())
                {
                    self.execute_layout_if_dirty(handle)?;
                    self.execute_scene_if_dirty(handle)?;
                }
                Ok(())
            }
            PlatformFact::Wake | PlatformFact::Exit => Ok(()),
            PlatformFact::PointerInput { handle, input } => {
                self.process_pointer_input(handle, input)
            }
            PlatformFact::KeyInput { handle, input } => self.process_key_input(handle, input),
            PlatformFact::TextInput { handle, input } => self.process_text_input(handle, input),
            PlatformFact::ImeInput { handle, input } => self.process_ime_input(handle, input),
            PlatformFact::ModifiersChanged { handle, modifiers } => {
                self.process_modifiers_changed(handle, modifiers)
            }
            PlatformFact::WheelInput { handle, input } => self.process_wheel_input(handle, input),
            PlatformFact::WindowFocusChanged { handle, input } => {
                self.process_window_focus_input(handle, input)
            }
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
                self.process_platform_fact(PlatformFact::CloseConfirmed { handle })?;
                self.process_platform_fact(PlatformFact::Destroyed { handle })
            }
            WindowCommand::Resize {
                handle,
                logical_size,
            } => self.apply_window_resize(handle, logical_size, false),
        }
    }

    fn apply_window_resize(
        &mut self,
        handle: AnyWindowHandle,
        logical_size: LayoutSize,
        defer_frame_work: bool,
    ) -> NekoResult<()> {
        self.state.resize_window(handle, logical_size)?;
        self.diagnostics
            .increment_signal(SignalId::SurfaceGenerationBumped);
        self.mark_size_or_scale_changed(handle);
        if !defer_frame_work {
            self.execute_layout_if_dirty(handle)?;
            self.execute_scene_if_dirty(handle)?;
        }
        Ok(())
    }

    fn mark_size_or_scale_changed(&mut self, handle: AnyWindowHandle) {
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Layout);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Semantics);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Surface);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Paint);
        self.request_redraw(handle.id());
    }

    fn destroy_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let _ = self.state.confirm_close_window(handle);
        self.state.close_window(handle)?;
        self.root_views.remove(&handle.id());
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Semantics);
        self.state
            .emit_retained_dirty(None, DirtyCause::WindowClosed, initial_lanes());
        self.diagnostics
            .increment_signal(SignalId::WindowLifecycleTransition);
        Ok(())
    }

    fn apply_renderability(
        &mut self,
        handle: AnyWindowHandle,
        renderability: Renderability,
    ) -> NekoResult<()> {
        let outcome = self.state.set_window_renderability(handle, renderability)?;
        self.diagnostics
            .increment_signal(SignalId::PlatformRenderabilityChanged);
        if !renderability.is_renderable() {
            self.diagnostics
                .increment_signal(SignalId::RuntimeNotRenderable);
        }
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Runtime,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "renderability.changed",
                "window renderability changed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("renderability", renderability.name()),
        );
        match outcome {
            RedrawRequestOutcome::Requested => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawRequested),
            RedrawRequestOutcome::Coalesced => {}
            RedrawRequestOutcome::SuppressedNotRenderable => self
                .diagnostics
                .increment_signal(SignalId::RuntimeRedrawSuppressed),
        }
        Ok(())
    }

    fn validate_platform_fact(&mut self, fact: &PlatformFact) -> NekoResult<()> {
        match fact {
            PlatformFact::WindowCreated { handle }
            | PlatformFact::WindowShown { handle }
            | PlatformFact::CloseRequested { handle }
            | PlatformFact::CloseConfirmed { handle }
            | PlatformFact::Destroyed { handle }
            | PlatformFact::RedrawRequested { handle } => self.validate_window(*handle),
            PlatformFact::LogicalSizeChanged {
                handle,
                logical_size,
            } => {
                self.validate_window(*handle)?;
                (*logical_size).validate_viewport()
            }
            PlatformFact::PhysicalSizeChanged { handle, .. } => self.validate_window(*handle),
            PlatformFact::ScaleFactorChanged {
                handle,
                scale_factor,
            } => {
                self.validate_window(*handle)?;
                if scale_factor.is_finite() && *scale_factor > 0.0 {
                    Ok(())
                } else {
                    Err(NekoError::invalid_input(
                        "viewport scale factor must be finite and positive",
                    ))
                }
            }
            PlatformFact::Minimized { handle }
            | PlatformFact::Restored { handle }
            | PlatformFact::RenderabilityChanged { handle, .. } => self.validate_window(*handle),
            PlatformFact::PointerInput { handle, input } => {
                self.validate_window(*handle)?;
                validate_pointer_input_position(*input)
            }
            PlatformFact::KeyInput { handle, .. }
            | PlatformFact::TextInput { handle, .. }
            | PlatformFact::ImeInput { handle, .. }
            | PlatformFact::ModifiersChanged { handle, .. }
            | PlatformFact::WindowFocusChanged { handle, .. } => self.validate_window(*handle),
            PlatformFact::WheelInput { handle, input } => {
                self.validate_window(*handle)?;
                validate_wheel_input(*input)
            }
            PlatformFact::Wake | PlatformFact::Exit => Ok(()),
        }
    }

    fn process_key_input(&mut self, handle: AnyWindowHandle, input: KeyInput) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        self.diagnostics.increment_signal(SignalId::InputKeyFact);
        self.record_key_input_fact(handle, &input);
        self.cleanup_stale_interaction_targets(handle)?;

        if input.synthetic() {
            self.record_key_dispatch(handle, &input, None, "synthetic_ignored", None);
            return Ok(());
        }

        let Some(target) = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.keyboard_focus())
        else {
            self.record_key_dispatch(handle, &input, None, "no_focus", None);
            return Ok(());
        };

        let Some(node) = self.current_target_node(handle, target)? else {
            self.state
                .interaction_mut(handle.id())
                .set_keyboard_focus(None);
            self.record_stale_input_target(handle, target, "keyboard_focus_dispatch");
            self.record_key_dispatch(handle, &input, Some(target), "stale", None);
            self.mark_semantic_interaction_changed(handle);
            return Ok(());
        };

        if !node.focusable() {
            self.state
                .interaction_mut(handle.id())
                .set_keyboard_focus(None);
            self.record_stale_input_target(handle, target, "keyboard_focus_not_focusable");
            self.record_focus_transition(
                handle,
                "keyboard",
                Some(target),
                None,
                "stale_not_focusable",
            );
            self.record_key_dispatch(handle, &input, Some(target), "stale", None);
            self.mark_semantic_interaction_changed(handle);
            return Ok(());
        }

        let handler = match input.kind() {
            KeyInputKind::Down => node.handlers().key_down(),
            KeyInputKind::Up => node.handlers().key_up(),
        };
        let Some(handler) = handler else {
            self.record_key_dispatch(handle, &input, Some(target), "no_handler", None);
            return self.apply_key_default_edit(handle, target, &input);
        };

        let event = KeyEvent::new(input.clone());
        let result = {
            let mut cx = AppContext::from_runtime(self);
            handler(&event, &mut cx)
        };
        let error_kind = result.as_ref().err().map(NekoError::kind);
        self.record_key_dispatch(
            handle,
            &input,
            Some(target),
            if error_kind.is_some() {
                "handler_error"
            } else {
                "dispatched"
            },
            error_kind,
        );
        result?;
        self.apply_key_default_edit(handle, target, &input)
    }

    fn apply_key_default_edit(
        &mut self,
        handle: AnyWindowHandle,
        keyboard_target: InteractionTarget,
        input: &KeyInput,
    ) -> NekoResult<()> {
        if !is_backspace_down(input) {
            return Ok(());
        }
        let Some(text_target) = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus())
        else {
            return Ok(());
        };
        if text_target != keyboard_target {
            return Ok(());
        }
        self.apply_delete_backward(handle, text_target)
    }

    fn process_text_input(&mut self, handle: AnyWindowHandle, input: TextInput) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        self.cleanup_stale_interaction_targets(handle)?;
        self.record_text_input_fact(handle, "text_input", input.text().len(), input.replace());
        let Some(target) = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus())
        else {
            self.record_text_edit(
                handle,
                None,
                "text_input",
                "no_focus",
                input.text().len(),
                None,
            );
            return Ok(());
        };
        self.apply_text_input(handle, target, &input, "text_input")
    }

    fn process_ime_input(&mut self, handle: AnyWindowHandle, input: ImeInput) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        self.diagnostics.increment_signal(SignalId::ImeTransition);
        self.cleanup_stale_interaction_targets(handle)?;
        match input {
            ImeInput::Enabled => {
                self.record_ime_transition(handle, "enabled", None, "accepted");
                Ok(())
            }
            ImeInput::Preedit(preedit) => {
                self.diagnostics.increment_signal(SignalId::ImePreedit);
                self.record_text_input_fact(
                    handle,
                    "preedit",
                    preedit.text().len(),
                    preedit.replace(),
                );
                let Some(target) = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.text_input_focus())
                else {
                    self.record_text_edit(
                        handle,
                        None,
                        "preedit",
                        "no_focus",
                        preedit.text().len(),
                        None,
                    );
                    return Ok(());
                };
                self.apply_ime_preedit(handle, target, &preedit)
            }
            ImeInput::Commit(input) => {
                self.diagnostics.increment_signal(SignalId::ImeCommit);
                self.record_text_input_fact(
                    handle,
                    "ime_commit",
                    input.text().len(),
                    input.replace(),
                );
                let Some(target) = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.text_input_focus())
                else {
                    self.record_text_edit(
                        handle,
                        None,
                        "ime_commit",
                        "no_focus",
                        input.text().len(),
                        None,
                    );
                    return Ok(());
                };
                self.apply_text_commit(handle, target, &input)
            }
            ImeInput::Disabled => {
                let target = self
                    .state
                    .interaction(handle.id())
                    .and_then(|state| state.text_input_focus());
                self.record_ime_transition(handle, "disabled", target, "accepted");
                let mut text_or_layout_changed = false;
                if let Some(target) = target
                    && matches!(
                        self.state
                            .retained_tree_mut(handle.id())
                            .and_then(|tree| tree.clear_composition_at_target(target)),
                        Some(TextEditOutcome::Mutated)
                    )
                {
                    self.mark_text_target_changed(handle, target);
                    text_or_layout_changed = true;
                }
                if target.is_some() {
                    self.state.push_ime_request(
                        handle.id(),
                        ImePlatformRequest::Allowed { allowed: true },
                    );
                    if !text_or_layout_changed {
                        self.refresh_text_input_cursor_area(handle);
                    }
                }
                Ok(())
            }
        }
    }

    fn record_text_input_fact(
        &mut self,
        handle: AnyWindowHandle,
        kind: &'static str,
        text_len: usize,
        replace: Option<TextRange>,
    ) {
        self.diagnostics.increment_signal(SignalId::TextInputFact);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "text.input_fact",
                "text input fact processed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("kind", kind)
            .with_field("text_len", text_len.to_string())
            .with_field("replace", format_text_range(replace)),
        );
    }

    fn record_text_edit(
        &mut self,
        handle: AnyWindowHandle,
        target: Option<InteractionTarget>,
        kind: &'static str,
        result: &'static str,
        text_len: usize,
        error_kind: Option<ErrorKind>,
    ) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Text,
                DiagnosticSeverity::Info,
                error_kind.unwrap_or(ErrorKind::Diagnostic),
                "text.edit",
                "editable text state update routed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("kind", kind)
            .with_field("result", result)
            .with_field("text_len", text_len.to_string())
            .with_field("target_id", format_target_id(target))
            .with_field("target_generation", format_target_generation(target))
            .with_field(
                "error_kind",
                error_kind.map_or_else(|| "none".to_owned(), |kind| format!("{kind:?}")),
            ),
        );
    }

    fn record_ime_transition(
        &mut self,
        handle: AnyWindowHandle,
        transition: &'static str,
        target: Option<InteractionTarget>,
        result: &'static str,
    ) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "ime.transition",
                "IME lifecycle transition routed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("transition", transition)
            .with_field("result", result)
            .with_field("target_id", format_target_id(target))
            .with_field("target_generation", format_target_generation(target)),
        );
    }

    fn record_ime_cursor_area(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        rect: LayoutRect,
    ) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "ime.cursor_area",
                "IME candidate cursor area refreshed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("target_id", target.node_id().raw().to_string())
            .with_field(
                "target_generation",
                target.node_generation().raw().to_string(),
            )
            .with_field("rect_x", rect.x().to_string())
            .with_field("rect_y", rect.y().to_string())
            .with_field("rect_width", rect.width().to_string())
            .with_field("rect_height", rect.height().to_string()),
        );
    }

    fn apply_text_commit(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        input: &TextInput,
    ) -> NekoResult<()> {
        self.apply_text_input(handle, target, input, "commit")
    }

    fn apply_text_input(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        input: &TextInput,
        kind: &'static str,
    ) -> NekoResult<()> {
        let result = self
            .state
            .retained_tree_mut(handle.id())
            .ok_or_else(|| NekoError::diagnostic("retained tree is missing for live window"))?
            .insert_text_at_target(target, input.text(), input.replace());
        self.finish_text_edit(handle, target, kind, input.text().len(), result)
    }

    fn apply_delete_backward(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
    ) -> NekoResult<()> {
        let result = self
            .state
            .retained_tree_mut(handle.id())
            .ok_or_else(|| NekoError::diagnostic("retained tree is missing for live window"))?
            .delete_backward_at_target(target);
        self.finish_text_edit(handle, target, "delete_backward", 0, result)
    }

    fn apply_ime_preedit(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        input: &crate::interaction::ImePreeditInput,
    ) -> NekoResult<()> {
        let result = self
            .state
            .retained_tree_mut(handle.id())
            .ok_or_else(|| NekoError::diagnostic("retained tree is missing for live window"))?
            .set_composition_at_target(target, input.text(), input.cursor(), input.replace());
        self.finish_text_edit(handle, target, "preedit", input.text().len(), result)
    }

    fn finish_text_edit(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        kind: &'static str,
        text_len: usize,
        result: Result<Option<TextEditOutcome>, TextRangeError>,
    ) -> NekoResult<()> {
        match result {
            Ok(Some(TextEditOutcome::Mutated)) => {
                self.mark_text_target_changed(handle, target);
                self.record_text_edit(handle, Some(target), kind, "mutated", text_len, None);
            }
            Ok(Some(TextEditOutcome::Unchanged)) => {
                self.record_text_edit(handle, Some(target), kind, "unchanged", text_len, None);
            }
            Ok(None) => {
                self.record_text_edit(handle, Some(target), kind, "stale", text_len, None);
                self.record_stale_input_target(handle, target, "text_edit_target");
            }
            Err(error) => {
                self.record_text_edit(
                    handle,
                    Some(target),
                    kind,
                    text_range_error_name(error),
                    text_len,
                    Some(ErrorKind::InvalidInput),
                );
            }
        }
        Ok(())
    }

    fn clear_text_input_focus(
        &mut self,
        handle: AnyWindowHandle,
        reason: &'static str,
    ) -> NekoResult<()> {
        let previous = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus());
        let Some(previous_target) = previous else {
            return Ok(());
        };
        if matches!(
            self.state
                .retained_tree_mut(handle.id())
                .and_then(|tree| tree.clear_composition_at_target(previous_target)),
            Some(TextEditOutcome::Mutated)
        ) {
            self.mark_text_target_changed(handle, previous_target);
        }
        self.state
            .interaction_mut(handle.id())
            .set_text_input_focus(None);
        self.state
            .push_ime_request(handle.id(), ImePlatformRequest::Allowed { allowed: false });
        self.record_focus_transition(handle, "text_input", previous, None, reason);
        self.mark_text_input_focus_changed(handle);
        Ok(())
    }

    fn focus_text_input_target(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        reason: &'static str,
    ) -> NekoResult<()> {
        let previous = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus());
        if previous == Some(target) {
            self.refresh_text_input_cursor_area(handle);
            return Ok(());
        }
        if let Some(previous_target) = previous
            && matches!(
                self.state
                    .retained_tree_mut(handle.id())
                    .and_then(|tree| tree.clear_composition_at_target(previous_target)),
                Some(TextEditOutcome::Mutated)
            )
        {
            self.mark_text_target_changed(handle, previous_target);
        }
        self.state
            .interaction_mut(handle.id())
            .set_text_input_focus(Some(target));
        self.state
            .push_ime_request(handle.id(), ImePlatformRequest::Allowed { allowed: true });
        self.state.push_ime_request(
            handle.id(),
            ImePlatformRequest::Purpose {
                purpose: TextInputPurpose::Normal,
            },
        );
        self.refresh_text_input_cursor_area(handle);
        self.record_focus_transition(handle, "text_input", previous, Some(target), reason);
        self.mark_text_input_focus_changed(handle);
        Ok(())
    }

    fn refresh_text_input_cursor_area(&mut self, handle: AnyWindowHandle) {
        let Some(target) = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus())
        else {
            return;
        };
        if let Some(rect) = self.text_input_cursor_area(handle, target) {
            self.state.replace_ime_candidate_rect(handle.id(), rect);
            self.record_ime_cursor_area(handle, target, rect);
        }
    }

    fn text_input_cursor_area(
        &self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
    ) -> Option<crate::layout::LayoutRect> {
        let retained = self.state.retained_snapshot(handle.id())?;
        let retained_node = find_node_by_target(&retained, target)?;
        if retained_node.kind() != ElementKind::Input {
            return None;
        }
        let layout = self.state.layout_snapshot(handle.id())?;
        let (layout_node, scroll_offset) = layout.root().and_then(|layout_root| {
            retained.root().and_then(|retained_root| {
                find_layout_node_by_target_with_scroll(
                    retained_root,
                    layout_root,
                    target,
                    LayoutPoint::ZERO,
                    self.state.interaction(handle.id()),
                )
            })
        })?;
        let content = layout_node.content_rect();
        let text_layout = layout_node.text_layout()?;
        Some(
            text_viewport_placement(retained_node.kind(), content, text_layout)
                .visible_caret_rect()
                .translate(-scroll_offset.x(), -scroll_offset.y()),
        )
    }

    fn mark_text_target_changed(&mut self, handle: AnyWindowHandle, target: InteractionTarget) {
        let mut lanes = DirtyLanes::empty();
        lanes.insert(DirtyLane::Text.flag());
        lanes.insert(DirtyLane::Layout.flag());
        lanes.insert(DirtyLane::Semantics.flag());
        lanes.insert(DirtyLane::Paint.flag());
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Text);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Layout);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Semantics);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Paint);
        self.state.emit_retained_dirty(
            Some(crate::retained::RetainedIdentity::new(
                target.node_id(),
                target.node_generation(),
            )),
            DirtyCause::TextChanged,
            lanes,
        );
        self.request_redraw(handle.id());
    }

    fn process_modifiers_changed(
        &mut self,
        handle: AnyWindowHandle,
        modifiers: Modifiers,
    ) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        self.state
            .interaction_mut(handle.id())
            .set_modifiers(modifiers);
        self.diagnostics
            .increment_signal(SignalId::InputModifiersFact);
        self.record_modifiers_fact(handle, modifiers);
        Ok(())
    }

    fn process_wheel_input(
        &mut self,
        handle: AnyWindowHandle,
        input: WheelInput,
    ) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        validate_wheel_input(input)?;
        self.diagnostics.increment_signal(SignalId::InputWheelFact);
        self.record_wheel_input_fact(handle, input);
        self.cleanup_stale_interaction_targets(handle)?;

        let Some(position) = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.last_hover_position())
        else {
            self.record_scroll_intent(handle, input, None, "miss", "no_hover_position");
            return Ok(());
        };
        let Some(entry) = self.hit_entry_at(handle, position) else {
            self.record_scroll_intent(handle, input, None, "miss", "hit_test_miss");
            return Ok(());
        };
        let Some(target) = self.nearest_scroll_target(handle, &entry)? else {
            self.record_scroll_intent(handle, input, None, "no_scroll_target", "not_scrollable");
            return Ok(());
        };
        self.record_scroll_intent(handle, input, Some(target), "accepted", "wheel_default");

        let (old_offset, new_offset, max_offset, unclamped_offset, changed) = {
            let interaction = self
                .state
                .interaction(handle.id())
                .cloned()
                .unwrap_or_default();
            let old_offset = interaction.scroll_offset(target);
            let max_offset = self
                .layout_scroll_geometry(handle, target)?
                .map_or(LayoutPoint::ZERO, |geometry| geometry.max_offset());
            let delta = wheel_delta_pixels(input);
            let unclamped_offset =
                LayoutPoint::new(old_offset.x() + delta.x(), old_offset.y() + delta.y());
            let new_offset = clamp_scroll_offset(unclamped_offset, max_offset);
            let changed = new_offset != old_offset;
            (
                old_offset,
                new_offset,
                max_offset,
                unclamped_offset,
                changed,
            )
        };
        self.record_scroll_clamp(handle, target, unclamped_offset, max_offset, new_offset);
        if changed {
            self.state
                .interaction_mut(handle.id())
                .set_scroll_offset(target, new_offset);
            self.refresh_text_input_cursor_area(handle);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Paint);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Semantics);
            self.request_redraw(handle.id());
        }
        self.record_scroll_offset(handle, target, old_offset, new_offset, max_offset, changed);
        Ok(())
    }

    fn apply_scroll_offset_clamps(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let targets = self
            .state
            .interaction(handle.id())
            .map(|interaction| interaction.scroll_offsets().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut any_changed = false;
        for (target, old_offset) in targets {
            let Some(geometry) = self.layout_scroll_geometry(handle, target)? else {
                continue;
            };
            let max_offset = geometry.max_offset();
            let new_offset = clamp_scroll_offset(old_offset, max_offset);
            if new_offset != old_offset {
                self.state
                    .interaction_mut(handle.id())
                    .set_scroll_offset(target, new_offset);
                self.record_scroll_clamp(handle, target, old_offset, max_offset, new_offset);
                self.record_scroll_offset(handle, target, old_offset, new_offset, max_offset, true);
                any_changed = true;
            }
        }
        if any_changed {
            self.refresh_text_input_cursor_area(handle);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Paint);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Semantics);
            self.request_redraw(handle.id());
        }
        Ok(())
    }

    fn process_window_focus_input(
        &mut self,
        handle: AnyWindowHandle,
        input: WindowFocusInput,
    ) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let previous = self
            .state
            .interaction(handle.id())
            .is_some_and(|state| state.window_focused());
        self.state
            .interaction_mut(handle.id())
            .set_window_focused(input.focused());
        self.diagnostics.increment_signal(SignalId::WindowFocusFact);
        self.record_window_focus_fact(handle, input);
        if previous != input.focused() {
            self.record_window_focus_transition(handle, previous, input.focused());
            self.mark_semantic_interaction_changed(handle);
        }
        if !input.focused() {
            self.clear_text_input_focus(handle, "window_unfocused")?;
        }
        Ok(())
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
                let previous = self.state.interaction(handle.id()).cloned();
                self.state.interaction_mut(handle.id()).set_hover(target);
                self.state
                    .interaction_mut(handle.id())
                    .set_last_hover_position(Some(input.position()));
                if let Err(error) = self.dispatch_pointer(handle, input, target, "pointer_move") {
                    let previous_hover = previous.as_ref().and_then(|state| state.hover());
                    let previous_position = previous.and_then(|state| state.last_hover_position());
                    self.state
                        .interaction_mut(handle.id())
                        .set_hover(previous_hover);
                    self.state
                        .interaction_mut(handle.id())
                        .set_last_hover_position(previous_position);
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
                self.apply_pointer_down_focus_default(handle, target)?;
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
        let target = self
            .hit_entry_at(handle, input.position())
            .map(|entry| InteractionTarget::new(entry.node_id(), entry.node_generation()));
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

    fn hit_entry_at(&self, handle: AnyWindowHandle, position: LayoutPoint) -> Option<HitTestEntry> {
        self.state
            .scene_snapshot(handle.id())
            .and_then(|scene| scene.hit_test().hit_test(position).cloned())
    }

    fn nearest_scroll_target(
        &mut self,
        handle: AnyWindowHandle,
        entry: &HitTestEntry,
    ) -> NekoResult<Option<InteractionTarget>> {
        for path_node in entry.path().iter().rev() {
            let target = InteractionTarget::new(path_node.node_id(), path_node.node_generation());
            let Some(node) = self.current_target_node(handle, target)? else {
                self.record_stale_input_target(handle, target, "scroll_path_cleanup");
                continue;
            };
            if node.resolved_style().layout().overflow() == crate::style::Overflow::Scroll
                && self
                    .layout_scroll_geometry(handle, target)?
                    .is_some_and(|geometry| geometry.scrollable())
            {
                return Ok(Some(target));
            }
        }
        Ok(None)
    }

    fn layout_scroll_geometry(
        &self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
    ) -> NekoResult<Option<crate::layout::ScrollGeometry>> {
        let layout = self.layout_snapshot(handle)?;
        Ok(layout
            .root()
            .and_then(|root| find_layout_node_by_target(root, target))
            .map(|node| node.scroll()))
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

    fn apply_pointer_down_focus_default(
        &mut self,
        handle: AnyWindowHandle,
        target: Option<InteractionTarget>,
    ) -> NekoResult<()> {
        let Some(target) = target else {
            self.clear_text_input_focus(handle, "pointer_down_miss")?;
            return Ok(());
        };
        let Some(node) = self.current_target_node(handle, target)? else {
            self.record_stale_input_target(handle, target, "keyboard_focus_default");
            self.clear_text_input_focus(handle, "pointer_down_stale")?;
            return Ok(());
        };
        if !node.focusable() {
            self.clear_text_input_focus(handle, "pointer_down_non_focusable")?;
            return Ok(());
        }
        let previous = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.keyboard_focus());
        if previous != Some(target) {
            self.state
                .interaction_mut(handle.id())
                .set_keyboard_focus(Some(target));
            self.record_focus_transition(
                handle,
                "keyboard",
                previous,
                Some(target),
                "pointer_down_default",
            );
            self.mark_semantic_interaction_changed(handle);
        }
        if node.kind() == ElementKind::Input {
            self.focus_text_input_target(handle, target, "pointer_down_default")?;
        } else {
            self.clear_text_input_focus(handle, "pointer_down_non_text_input")?;
        }
        Ok(())
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
        let keyboard_focus = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.keyboard_focus());
        let text_input_focus = self
            .state
            .interaction(handle.id())
            .and_then(|state| state.text_input_focus());
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
        if let Some(target) = keyboard_focus
            && find_focusable_node_by_target(&retained, target).is_none()
        {
            self.state
                .interaction_mut(handle.id())
                .set_keyboard_focus(None);
            self.record_stale_input_target(handle, target, "keyboard_focus_cleanup");
            self.record_focus_transition(handle, "keyboard", Some(target), None, "stale_cleanup");
            self.mark_semantic_interaction_changed(handle);
        }
        if let Some(target) = text_input_focus
            && find_text_input_node_by_target(&retained, target).is_none()
        {
            self.record_stale_input_target(handle, target, "text_input_focus_cleanup");
            self.clear_text_input_focus(handle, "stale_cleanup")?;
        }
        let stale_scroll_targets = self
            .state
            .interaction_mut(handle.id())
            .retain_scroll_offsets(|target| {
                find_scrollable_node_by_target(&retained, target).is_some()
            });
        let removed_scroll_offsets = !stale_scroll_targets.is_empty();
        for target in stale_scroll_targets {
            self.record_stale_input_target(handle, target, "scroll_offset_cleanup");
        }
        if removed_scroll_offsets {
            self.mark_semantic_interaction_changed(handle);
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
                self.refresh_text_input_cursor_area(handle);
                self.apply_scroll_offset_clamps(handle)?;
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
        self.execute_semantics_if_dirty(handle)?;
        if !self.state.window(handle)?.renderability().is_renderable() {
            self.diagnostics
                .increment_signal(SignalId::RuntimeNotRenderable);
            return Ok(());
        }
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
        let expected_generation = scene_generation_for_inputs_with_interaction(
            &retained,
            &style,
            &layout,
            self.state.interaction(handle.id()),
        );
        let previous = self.state.scene_snapshot(handle.id());
        let output = compile_scene(SceneCompileInput {
            retained: &retained,
            style: &style,
            layout: &layout,
            interaction: self.state.interaction(handle.id()),
            previous: previous.as_ref(),
        });
        let current_retained = self.retained_snapshot(handle)?;
        let current_style = self.style_snapshot(handle)?;
        let current_layout = self.layout_snapshot(handle)?;
        if !scene_publish_is_current_with_interaction(
            expected_generation,
            output.scene.generation(),
            &current_retained,
            &current_style,
            &current_layout,
            self.state.interaction(handle.id()),
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
        self.state
            .set_scene_snapshot(handle.id(), output.scene.clone());
        let record = self.state.window(handle)?.clone();
        let expected_surface_generation = record.surface_generation();
        let frame_context = PreparedFrameContext::for_surface(
            record.viewport(),
            record.physical_size(),
            expected_surface_generation,
        );
        let prepared_frame = prepare_frame_graph_for_surface(&output.scene, frame_context);
        let current_surface_generation = self.state.window(handle)?.surface_generation();
        if !prepared_frame
            .is_current_for_scene_and_surface(&output.scene, current_surface_generation)
        {
            let mut stats = prepared_frame.stats().clone();
            stats.stale_drop_count = stats.stale_drop_count.saturating_add(1);
            self.record_render_frame_graph(handle, &stats, "stale_drop");
            self.state.set_last_frame_graph(stats);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Paint);
            return Ok(());
        }
        let stats = prepared_frame.stats().clone();
        self.record_render_frame_graph(handle, &stats, "prepared");
        self.state.set_last_frame_graph(stats);
        self.state
            .set_prepared_frame_snapshot(handle.id(), prepared_frame);
        Ok(())
    }

    fn execute_semantics_if_dirty(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.state.validate_window(handle)?;
        let taken = self
            .state
            .scheduler_mut()
            .take_dirty_lanes(handle.id(), DirtyLane::Semantics.flag());
        if !taken.contains(DirtyLane::Semantics.flag()) {
            return Ok(());
        }
        self.state.record_consumed_dirty_lanes(handle.id(), taken);

        let retained = self.retained_snapshot(handle)?;
        let style = self.style_snapshot(handle)?;
        let layout = self.layout_snapshot(handle)?;
        let output = build_semantic_snapshot(SemanticBuildInput {
            retained: &retained,
            style: &style,
            layout: &layout,
            interaction: self.state.interaction(handle.id()),
        });
        let current_retained = self.retained_snapshot(handle)?;
        let current_style = self.style_snapshot(handle)?;
        let current_layout = self.layout_snapshot(handle)?;
        if !semantic_publish_is_current_with_interaction(
            output.snapshot.generation(),
            &current_retained,
            &current_style,
            &current_layout,
            self.state.interaction(handle.id()),
        ) {
            let mut stats = output.stats;
            stats.diagnostic_count = 0;
            stats.stale_drop_count = 1;
            self.record_semantic_build(handle, &stats, "stale_drop");
            self.state.set_last_semantic_build(stats);
            self.state
                .scheduler_mut()
                .mark_dirty(handle.id(), DirtyLane::Semantics);
            return Ok(());
        }

        let snapshot = output.snapshot;
        let stats = output.stats;
        self.record_semantic_build(handle, &stats, "published");
        self.diagnostics.extend_records(output.records);
        self.state.set_last_semantic_build(stats);
        self.state.set_semantic_snapshot(handle.id(), snapshot);
        Ok(())
    }

    fn execute_semantics_without_redraw_if_ready(&mut self) -> NekoResult<()> {
        for window in self.state.live_window_ids() {
            let Some(state) = self.state.scheduler().window_state(window) else {
                continue;
            };
            let dirty_lanes = state.dirty_lanes();
            if state.pending_redraw()
                || !dirty_lanes.contains(DirtyLane::Semantics.flag())
                || dirty_lanes.contains(DirtyLane::Layout.flag())
                || self.state.layout_snapshot(window).is_none()
            {
                continue;
            }
            let handle = self.state.window_by_id(window)?.handle();
            self.execute_semantics_if_dirty(handle)?;
        }
        Ok(())
    }

    fn record_semantic_build(
        &mut self,
        handle: AnyWindowHandle,
        stats: &SemanticBuildStats,
        result: &'static str,
    ) {
        self.semantic_build_duration += stats.duration;
        self.diagnostics.increment_signal(SignalId::SemanticsBuild);
        self.diagnostics
            .add_signal(SignalId::SemanticsNodeCount, stats.node_count as u64);
        self.diagnostics
            .add_signal(SignalId::SemanticsDiagnostic, stats.diagnostic_count as u64);
        self.diagnostics
            .add_signal(SignalId::SemanticsStaleDrop, stats.stale_drop_count);
        self.diagnostics.add_signal(
            SignalId::SemanticsDurationMicros,
            duration_micros_signal(stats.duration),
        );
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Semantics,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "semantics.build",
                "semantic snapshot build completed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("result", result)
            .with_field("node_count", stats.node_count.to_string())
            .with_field("diagnostic_count", stats.diagnostic_count.to_string())
            .with_field("stale_drop_count", stats.stale_drop_count.to_string())
            .with_field("duration_micros", stats.duration.as_micros().to_string()),
        );
    }

    fn record_render_frame_graph(
        &mut self,
        handle: AnyWindowHandle,
        stats: &FrameGraphStats,
        result: &'static str,
    ) {
        self.render_prepare_duration += stats.duration;
        self.diagnostics
            .increment_signal(SignalId::RenderFrameGraph);
        self.diagnostics
            .add_signal(SignalId::RenderPass, stats.pass_count as u64);
        self.diagnostics
            .add_signal(SignalId::RenderUploadPlan, stats.upload_intent_count as u64);
        self.diagnostics
            .add_signal(SignalId::RenderLayer, stats.layer_count as u64);
        self.diagnostics
            .add_signal(SignalId::RenderStaleDrop, stats.stale_drop_count);
        self.diagnostics.add_signal(
            SignalId::RenderUnsupported,
            stats.unsupported_fragment_count as u64,
        );
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Render,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "render.frame_graph",
                "render frame graph prepared",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("result", result)
            .with_field(
                "surface_generation",
                stats
                    .surface_generation
                    .map_or_else(|| "none".to_owned(), |generation| generation.to_string()),
            )
            .with_field("pass_count", stats.pass_count.to_string())
            .with_field("draw_item_count", stats.draw_item_count.to_string())
            .with_field("upload_intent_count", stats.upload_intent_count.to_string())
            .with_field("layer_count", stats.layer_count.to_string())
            .with_field("box_shape_count", stats.box_shape_count.to_string())
            .with_field(
                "unsupported_fragment_count",
                stats.unsupported_fragment_count.to_string(),
            )
            .with_field("duration_micros", stats.duration.as_micros().to_string()),
        );
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

    fn record_key_input_fact(&mut self, handle: AnyWindowHandle, input: &KeyInput) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "input.key_fact",
                "key input fact processed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("kind", key_input_kind_name(input.kind()))
            .with_field("logical_kind", input.logical_key().kind_name())
            .with_field("logical_key", input.logical_key().name().to_owned())
            .with_field("physical_key", input.physical_key().name().to_owned())
            .with_field("repeat", input.repeat().to_string())
            .with_field("synthetic", input.synthetic().to_string())
            .with_field("modifiers_bits", input.modifiers().bits().to_string())
            .with_field("shift", input.modifiers().shift().to_string())
            .with_field("ctrl", input.modifiers().ctrl().to_string())
            .with_field("alt", input.modifiers().alt().to_string())
            .with_field("logo", input.modifiers().logo().to_string()),
        );
    }

    fn record_modifiers_fact(&mut self, handle: AnyWindowHandle, modifiers: Modifiers) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "input.modifiers_fact",
                "keyboard modifiers fact processed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("modifiers_bits", modifiers.bits().to_string())
            .with_field("shift", modifiers.shift().to_string())
            .with_field("ctrl", modifiers.ctrl().to_string())
            .with_field("alt", modifiers.alt().to_string())
            .with_field("logo", modifiers.logo().to_string())
            .with_field("command", modifiers.command().to_string()),
        );
    }

    fn record_wheel_input_fact(&mut self, handle: AnyWindowHandle, input: WheelInput) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "input.wheel_fact",
                "wheel input fact processed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("delta_unit", input.delta().unit_name())
            .with_field("delta_x", input.delta().x().to_string())
            .with_field("delta_y", input.delta().y().to_string())
            .with_field("phase", input.phase().name())
            .with_field("modifiers_bits", input.modifiers().bits().to_string()),
        );
    }

    fn record_scroll_intent(
        &mut self,
        handle: AnyWindowHandle,
        input: WheelInput,
        target: Option<InteractionTarget>,
        result: &'static str,
        reason: &'static str,
    ) {
        self.diagnostics.increment_signal(SignalId::ScrollIntent);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "scroll.intent",
                "wheel scroll intent routed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("result", result)
            .with_field("reason", reason)
            .with_field("delta_unit", input.delta().unit_name())
            .with_field("delta_x", input.delta().x().to_string())
            .with_field("delta_y", input.delta().y().to_string())
            .with_field("phase", input.phase().name())
            .with_field("target_id", format_target_id(target))
            .with_field("target_generation", format_target_generation(target)),
        );
    }

    fn record_scroll_offset(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        old_offset: LayoutPoint,
        new_offset: LayoutPoint,
        max_offset: LayoutPoint,
        changed: bool,
    ) {
        self.diagnostics.increment_signal(SignalId::ScrollOffset);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "scroll.offset",
                "scroll offset updated",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("target_id", target.node_id().raw().to_string())
            .with_field(
                "target_generation",
                target.node_generation().raw().to_string(),
            )
            .with_field("old_x", old_offset.x().to_string())
            .with_field("old_y", old_offset.y().to_string())
            .with_field("new_x", new_offset.x().to_string())
            .with_field("new_y", new_offset.y().to_string())
            .with_field("max_x", max_offset.x().to_string())
            .with_field("max_y", max_offset.y().to_string())
            .with_field("changed", changed.to_string()),
        );
    }

    fn record_scroll_clamp(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        unclamped_offset: LayoutPoint,
        max_offset: LayoutPoint,
        clamped_offset: LayoutPoint,
    ) {
        self.diagnostics.increment_signal(SignalId::ScrollClamp);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "scroll.clamp",
                "scroll offset clamped to layout extent",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("target_id", target.node_id().raw().to_string())
            .with_field(
                "target_generation",
                target.node_generation().raw().to_string(),
            )
            .with_field("unclamped_x", unclamped_offset.x().to_string())
            .with_field("unclamped_y", unclamped_offset.y().to_string())
            .with_field("max_x", max_offset.x().to_string())
            .with_field("max_y", max_offset.y().to_string())
            .with_field("clamped_x", clamped_offset.x().to_string())
            .with_field("clamped_y", clamped_offset.y().to_string()),
        );
    }

    fn record_window_focus_fact(&mut self, handle: AnyWindowHandle, input: WindowFocusInput) {
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Window,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "window.focus_fact",
                "window focus fact processed through runtime",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("focused", input.focused().to_string()),
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

    fn record_key_dispatch(
        &mut self,
        handle: AnyWindowHandle,
        input: &KeyInput,
        target: Option<InteractionTarget>,
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
                "key input dispatched",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("raw_kind", key_input_kind_name(input.kind()))
            .with_field("event_kind", key_event_kind_name(input.kind()))
            .with_field("result", result)
            .with_field(
                "error_kind",
                error_kind.map_or_else(|| "none".to_owned(), |kind| format!("{kind:?}")),
            )
            .with_field("logical_kind", input.logical_key().kind_name())
            .with_field("logical_key", input.logical_key().name().to_owned())
            .with_field("physical_key", input.physical_key().name().to_owned())
            .with_field("repeat", input.repeat().to_string())
            .with_field("synthetic", input.synthetic().to_string())
            .with_field("modifiers_bits", input.modifiers().bits().to_string())
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

    fn record_window_focus_transition(
        &mut self,
        handle: AnyWindowHandle,
        previous: bool,
        current: bool,
    ) {
        self.diagnostics.increment_signal(SignalId::FocusTransition);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "focus.transition",
                "focus target changed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("domain", "window")
            .with_field("from", focus_bool_name(previous))
            .with_field("to", focus_bool_name(current))
            .with_field(
                "reason",
                if current {
                    "window_focused"
                } else {
                    "window_unfocused"
                },
            )
            .with_field("generation", handle.generation().raw().to_string()),
        );
    }

    fn record_focus_transition(
        &mut self,
        handle: AnyWindowHandle,
        domain: &'static str,
        from: Option<InteractionTarget>,
        to: Option<InteractionTarget>,
        reason: &'static str,
    ) {
        self.diagnostics.increment_signal(SignalId::FocusTransition);
        self.diagnostics.record(
            DiagnosticRecord::new(
                DiagnosticArea::Input,
                DiagnosticSeverity::Info,
                ErrorKind::Diagnostic,
                "focus.transition",
                "focus target changed",
            )
            .with_field("window", handle.id().raw().to_string())
            .with_field("domain", domain)
            .with_field("from", format_focus_target(from))
            .with_field("to", format_focus_target(to))
            .with_field("reason", reason)
            .with_field("generation", focus_transition_generation(to.or(from))),
        );
    }

    fn record_stale_input_target(
        &mut self,
        handle: AnyWindowHandle,
        target: InteractionTarget,
        state_kind: &'static str,
    ) {
        let actual_generation = self
            .state
            .retained_snapshot(handle.id())
            .and_then(|retained| find_node_by_id(&retained, target).map(|node| node.generation()));
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
            )
            .with_field(
                "actual_generation",
                actual_generation.map_or_else(
                    || "none".to_owned(),
                    |generation| generation.raw().to_string(),
                ),
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

    fn mark_semantic_interaction_changed(&mut self, handle: AnyWindowHandle) {
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Semantics);
    }

    fn mark_text_input_focus_changed(&mut self, handle: AnyWindowHandle) {
        self.mark_semantic_interaction_changed(handle);
        self.state
            .scheduler_mut()
            .mark_dirty(handle.id(), DirtyLane::Paint);
        self.request_redraw(handle.id());
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

fn key_input_kind_name(kind: KeyInputKind) -> &'static str {
    match kind {
        KeyInputKind::Down => "down",
        KeyInputKind::Up => "up",
    }
}

fn key_event_kind_name(kind: KeyInputKind) -> &'static str {
    match kind {
        KeyInputKind::Down => "key_down",
        KeyInputKind::Up => "key_up",
    }
}

fn is_backspace_down(input: &KeyInput) -> bool {
    input.kind() == KeyInputKind::Down
        && matches!(input.logical_key(), Key::Named(name) if name == "Backspace")
}

fn focus_bool_name(focused: bool) -> &'static str {
    if focused { "focused" } else { "unfocused" }
}

fn format_focus_target(target: Option<InteractionTarget>) -> String {
    target.map_or_else(
        || "none".to_owned(),
        |target| {
            format!(
                "{}@{}",
                target.node_id().raw(),
                target.node_generation().raw()
            )
        },
    )
}

fn focus_transition_generation(target: Option<InteractionTarget>) -> String {
    target.map_or_else(
        || "none".to_owned(),
        |target| target.node_generation().raw().to_string(),
    )
}

fn format_target_id(target: Option<InteractionTarget>) -> String {
    target.map_or_else(
        || "none".to_owned(),
        |target| target.node_id().raw().to_string(),
    )
}

fn format_target_generation(target: Option<InteractionTarget>) -> String {
    target.map_or_else(
        || "none".to_owned(),
        |target| target.node_generation().raw().to_string(),
    )
}

fn format_text_range(range: Option<TextRange>) -> String {
    range.map_or_else(
        || "none".to_owned(),
        |range| format!("{}..{}", range.start(), range.end()),
    )
}

fn text_range_error_name(error: TextRangeError) -> &'static str {
    match error {
        TextRangeError::Reversed => "invalid_reversed_range",
        TextRangeError::OutOfBounds => "invalid_out_of_bounds_range",
        TextRangeError::NotBoundary => "invalid_char_boundary_range",
    }
}

fn wheel_delta_pixels(input: WheelInput) -> LayoutPoint {
    const LINE_HEIGHT: f32 = 40.0;
    match input.delta() {
        ScrollDelta::Lines { x, y } => LayoutPoint::new(x * LINE_HEIGHT, y * LINE_HEIGHT),
        ScrollDelta::Pixels { x, y } => LayoutPoint::new(x, y),
    }
}

fn clamp_scroll_offset(offset: LayoutPoint, max_offset: LayoutPoint) -> LayoutPoint {
    LayoutPoint::new(
        offset.x().clamp(0.0, max_offset.x()),
        offset.y().clamp(0.0, max_offset.y()),
    )
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

fn validate_wheel_input(input: WheelInput) -> NekoResult<()> {
    if input.delta().is_finite() {
        Ok(())
    } else {
        Err(NekoError::invalid_input(
            "wheel delta must be finite in both axes",
        ))
    }
}

fn find_node_by_target(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    retained
        .root()
        .and_then(|root| find_node_by_target_from(root, target))
}

fn find_focusable_node_by_target(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    find_node_by_target(retained, target).filter(|node| node.focusable())
}

fn find_scrollable_node_by_target(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    find_node_by_target(retained, target)
        .filter(|node| node.resolved_style().layout().overflow() == crate::style::Overflow::Scroll)
}

fn find_text_input_node_by_target(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    find_node_by_target(retained, target).filter(|node| node.kind() == ElementKind::Input)
}

fn find_layout_node_by_target(
    node: &LayoutNodeSnapshot,
    target: InteractionTarget,
) -> Option<&LayoutNodeSnapshot> {
    if node.node_id() == target.node_id() {
        return Some(node);
    }
    node.children()
        .iter()
        .find_map(|child| find_layout_node_by_target(child, target))
}

fn find_layout_node_by_target_with_scroll<'a>(
    retained: &RetainedNodeSnapshot,
    layout: &'a LayoutNodeSnapshot,
    target: InteractionTarget,
    scroll_offset: LayoutPoint,
    interaction: Option<&crate::interaction::InteractionState>,
) -> Option<(&'a LayoutNodeSnapshot, LayoutPoint)> {
    if retained.id() != layout.node_id() {
        return None;
    }
    if layout.node_id() == target.node_id() {
        return Some((layout, scroll_offset));
    }
    let node_target =
        crate::interaction::InteractionTarget::new(retained.id(), retained.generation());
    let child_scroll_offset = if layout.scroll().overflow() == crate::style::Overflow::Scroll {
        let current =
            interaction.map_or(LayoutPoint::ZERO, |state| state.scroll_offset(node_target));
        scroll_offset.translate(current.x(), current.y())
    } else {
        scroll_offset
    };
    retained.children().iter().find_map(|retained_child| {
        layout
            .children()
            .iter()
            .find(|layout_child| layout_child.node_id() == retained_child.id())
            .and_then(|layout_child| {
                find_layout_node_by_target_with_scroll(
                    retained_child,
                    layout_child,
                    target,
                    child_scroll_offset,
                    interaction,
                )
            })
    })
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

fn find_node_by_id(
    retained: &RetainedTreeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    retained
        .root()
        .and_then(|root| find_node_by_id_from(root, target))
}

fn find_node_by_id_from(
    node: &RetainedNodeSnapshot,
    target: InteractionTarget,
) -> Option<&RetainedNodeSnapshot> {
    if node.id() == target.node_id() {
        return Some(node);
    }
    node.children()
        .iter()
        .find_map(|child| find_node_by_id_from(child, target))
}
