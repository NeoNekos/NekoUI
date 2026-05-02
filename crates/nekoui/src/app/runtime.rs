use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{Receiver, unbounded};
use hashbrown::HashMap;

use crate::element::{AnyElement, BuildCx, BuildResult, IntoElement, SpecArena};
use crate::error::{Error, RuntimeError};
use crate::input::TextInputEvent;
use crate::platform::window::{WindowCommand, WindowCommandSender};
use crate::scene::DirtyLaneMask;
use crate::window::{DisplayId, DisplayInfo, WindowHandle, WindowId, WindowInfo, WindowOptions};

use super::context::{AppContext, Context};
use super::executor::{BackgroundExecutor, UiExecutor};
use super::handle::{Entity, View};
use super::{Render, WakeHandle, WindowBuildFn};

type ViewRenderer =
    fn(u64, &Rc<RefCell<RuntimeState>>, &WindowInfo) -> Result<AnyElement, RuntimeError>;
type ObserveCallback = Box<dyn FnMut(&Rc<RefCell<RuntimeState>>, u64) -> Result<(), RuntimeError>>;
type EventCallback =
    Box<dyn FnMut(&Rc<RefCell<RuntimeState>>, u64, &dyn Any) -> Result<(), RuntimeError>>;
type TextInputCallback =
    Box<dyn FnMut(&Rc<RefCell<RuntimeState>>, u64, &TextInputEvent) -> Result<(), RuntimeError>>;

pub(crate) struct App {
    runtime: Rc<RefCell<RuntimeState>>,
    pending_windows: VecDeque<PendingWindowRequest>,
    wake_handle: Option<WakeHandle>,
    background_executor: BackgroundExecutor,
    ui_executor: UiExecutor,
    window_commands: WindowCommandSender,
    window_command_receiver: Receiver<(WindowId, WindowCommand)>,
    displays: Vec<DisplayInfo>,
    active_display: Option<DisplayId>,
}

pub(in crate::app) struct RuntimeState {
    pub(in crate::app) background_executor: BackgroundExecutor,
    pub(in crate::app) ui_executor: UiExecutor,
    pub(in crate::app) entities: HashMap<u64, Box<dyn Any>>,
    pub(in crate::app) dirty_entities: HashMap<u64, DirtyLaneMask>,
    pub(in crate::app) view_renderers: HashMap<u64, ViewRenderer>,
    pub(in crate::app) observe_subscriptions: HashMap<u64, Vec<ObserveSubscription>>,
    pub(in crate::app) event_subscriptions: HashMap<EventSubscriptionKey, Vec<EventSubscription>>,
    pub(in crate::app) text_input_subscriptions: HashMap<u64, Vec<TextInputSubscription>>,
    pub(in crate::app) pending_events: VecDeque<QueuedEvent>,
}

pub(in crate::app) struct ObserveSubscription {
    pub(in crate::app) active: Arc<AtomicBool>,
    pub(in crate::app) callback: ObserveCallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::app) struct EventSubscriptionKey {
    pub(in crate::app) source_id: u64,
    pub(in crate::app) event_type: TypeId,
}

pub(in crate::app) struct EventSubscription {
    pub(in crate::app) active: Arc<AtomicBool>,
    pub(in crate::app) callback: EventCallback,
}

pub(in crate::app) struct TextInputSubscription {
    pub(in crate::app) active: Arc<AtomicBool>,
    pub(in crate::app) callback: TextInputCallback,
}

pub(in crate::app) struct QueuedEvent {
    pub(in crate::app) source_id: u64,
    pub(in crate::app) event_type: TypeId,
    pub(in crate::app) payload: Box<dyn Any>,
}

#[derive(Default)]
pub(crate) struct RuntimeProcessResult {
    pub(crate) dirty_views: HashMap<u64, DirtyLaneMask>,
}

pub(crate) struct PendingWindowRequest {
    pub(crate) id: WindowId,
    pub(crate) options: WindowOptions,
    pub(crate) build_root: WindowBuildFn,
}

impl App {
    pub(crate) fn new(displays: Vec<DisplayInfo>) -> Self {
        let background_executor = BackgroundExecutor::new();
        let ui_executor = UiExecutor::new();
        let (window_command_sender, window_command_receiver) = unbounded();
        let active_display = displays
            .iter()
            .find(|display| display.is_primary)
            .map(|display| display.id);
        Self {
            runtime: Rc::new(RefCell::new(RuntimeState {
                background_executor: background_executor.clone(),
                ui_executor: ui_executor.clone(),
                entities: HashMap::new(),
                dirty_entities: HashMap::new(),
                view_renderers: HashMap::new(),
                observe_subscriptions: HashMap::new(),
                event_subscriptions: HashMap::new(),
                text_input_subscriptions: HashMap::new(),
                pending_events: VecDeque::new(),
            })),
            pending_windows: VecDeque::new(),
            wake_handle: None,
            background_executor,
            ui_executor,
            window_commands: WindowCommandSender::new(window_command_sender),
            window_command_receiver,
            displays,
            active_display,
        }
    }

    pub(crate) fn open_window<V>(
        &mut self,
        options: WindowOptions,
        build_root: impl FnOnce(&WindowInfo, &mut AppContext<'_>) -> View<V> + 'static,
    ) -> Result<WindowHandle<V>, Error>
    where
        V: Render + 'static,
    {
        let (request, handle) = self.prepare_window_request(options, build_root);
        self.enqueue_window_request(request);
        Ok(handle)
    }

    pub(crate) fn prepare_window_request<V>(
        &mut self,
        options: WindowOptions,
        build_root: impl FnOnce(&WindowInfo, &mut AppContext<'_>) -> View<V> + 'static,
    ) -> (PendingWindowRequest, WindowHandle<V>)
    where
        V: Render + 'static,
    {
        let id = WindowId::new();
        let request = PendingWindowRequest {
            id,
            options,
            build_root: Box::new(move |window, app| {
                let mut cx = AppContext::new(app);
                build_root(window, &mut cx).into_any_element()
            }),
        };
        let handle = WindowHandle::new(id, self.window_commands.clone());
        (request, handle)
    }

    pub(crate) fn enqueue_window_request(&mut self, request: PendingWindowRequest) {
        self.pending_windows.push_back(request);
        self.wake_runtime();
    }

    pub(crate) fn insert_entity<T>(&self, state: T) -> Entity<T>
    where
        T: 'static,
    {
        let entity = Entity::new();
        self.runtime
            .borrow_mut()
            .entities
            .insert(entity.id(), Box::new(state));
        entity
    }

    pub(crate) fn insert_view<T>(&self, state: T) -> View<T>
    where
        T: Render + 'static,
    {
        let view = View::new();
        let mut runtime = self.runtime.borrow_mut();
        runtime.entities.insert(view.id(), Box::new(state));
        runtime
            .view_renderers
            .insert(view.id(), render_view_entity::<T>);
        view
    }

    pub(crate) fn update<T, R>(
        &self,
        entity: Entity<T>,
        updater: impl FnOnce(&mut T, &mut Context<'_, T>) -> R,
    ) -> Result<R, RuntimeError>
    where
        T: 'static,
    {
        let boxed = self
            .runtime
            .borrow_mut()
            .entities
            .remove(&entity.id())
            .ok_or(RuntimeError::EntityNotFound(entity.id()))?;

        let mut typed = boxed
            .downcast::<T>()
            .map_err(|_| RuntimeError::TypeMismatch(entity.id()))?;

        let (background_executor, ui_executor) = {
            let runtime = self.runtime.borrow();
            (
                runtime.background_executor.clone(),
                runtime.ui_executor.clone(),
            )
        };

        let result = {
            let mut cx = Context::new(
                entity,
                self.runtime.clone(),
                background_executor,
                ui_executor,
            );
            updater(typed.as_mut(), &mut cx)
        };

        self.runtime
            .borrow_mut()
            .entities
            .insert(entity.id(), typed as Box<dyn Any>);

        Ok(result)
    }

    pub(crate) fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    pub(crate) fn ui_executor(&self) -> UiExecutor {
        self.ui_executor.clone()
    }

    pub(crate) fn displays(&self) -> Vec<DisplayInfo> {
        self.displays.clone()
    }

    pub(crate) fn primary_display(&self) -> Option<DisplayInfo> {
        self.displays
            .iter()
            .find(|display| display.is_primary)
            .cloned()
    }

    pub(crate) fn active_display(&self) -> Option<DisplayInfo> {
        let active = self.active_display?;
        self.displays
            .iter()
            .find(|display| display.id == active)
            .cloned()
    }

    pub(crate) fn set_displays(&mut self, displays: Vec<DisplayInfo>) {
        self.displays = displays;
        if self
            .active_display
            .is_some_and(|active| self.displays.iter().all(|display| display.id != active))
        {
            self.active_display = self.primary_display().map(|display| display.id);
        }
    }

    pub(crate) fn set_active_display(&mut self, active_display: Option<DisplayId>) {
        self.active_display =
            active_display.or_else(|| self.primary_display().map(|display| display.id));
    }

    pub(crate) fn set_wake_handle(&mut self, wake_handle: Option<WakeHandle>) {
        self.wake_handle = wake_handle.clone();
        self.ui_executor.set_wake_handle(wake_handle);
    }

    pub(crate) fn drain_window_requests(&mut self) -> Vec<PendingWindowRequest> {
        self.pending_windows.drain(..).collect()
    }

    pub(crate) fn drain_window_commands(&mut self) -> Vec<(WindowId, WindowCommand)> {
        self.window_command_receiver.try_iter().collect()
    }

    pub(crate) fn process_runtime(&mut self) -> Result<RuntimeProcessResult, RuntimeError> {
        self.ui_executor.run_pending();

        let mut result = RuntimeProcessResult::default();

        loop {
            let dirty_batch = self.take_dirty_batch();
            let event = self.take_pending_event();

            if dirty_batch.is_empty() && event.is_none() {
                break;
            }

            for (entity_id, lane) in dirty_batch {
                if self
                    .runtime
                    .borrow()
                    .view_renderers
                    .contains_key(&entity_id)
                {
                    merge_lane(&mut result.dirty_views, entity_id, lane);
                }
                self.dispatch_observers(entity_id)?;
            }

            if let Some(event) = event {
                self.dispatch_event(event)?;
            }
        }

        Ok(result)
    }

    pub(crate) fn dispatch_text_input_event(
        &self,
        source_id: u64,
        event: &TextInputEvent,
    ) -> Result<(), RuntimeError> {
        let mut callbacks = {
            let mut runtime = self.runtime.borrow_mut();
            std::mem::take(runtime.text_input_subscriptions.entry(source_id).or_default())
        };

        for subscription in &mut callbacks {
            if subscription.active.load(Ordering::Relaxed) {
                (subscription.callback)(&self.runtime, source_id, event)?;
            }
        }

        callbacks.retain(|subscription| subscription.active.load(Ordering::Relaxed));
        if !callbacks.is_empty() {
            self.runtime
                .borrow_mut()
                .text_input_subscriptions
                .entry(source_id)
                .or_default()
                .append(&mut callbacks);
        }

        Ok(())
    }

    pub(crate) fn build_root_spec(
        &self,
        window: &WindowInfo,
        template: &AnyElement,
        arena: &mut SpecArena,
    ) -> Result<BuildResult, RuntimeError> {
        let runtime = self.runtime.clone();
        let mut resolver = move |entity_id: u64, window: &WindowInfo| {
            let renderer = runtime
                .borrow()
                .view_renderers
                .get(&entity_id)
                .copied()
                .ok_or(RuntimeError::EntityNotFound(entity_id))?;
            renderer(entity_id, &runtime, window)
        };

        BuildCx::new(window, &mut resolver, arena).build_root(template.clone())
    }

    fn wake_runtime(&self) {
        if let Some(wake_handle) = &self.wake_handle {
            wake_handle();
        }
    }

    fn take_dirty_batch(&self) -> Vec<(u64, DirtyLaneMask)> {
        self.runtime
            .borrow_mut()
            .dirty_entities
            .drain()
            .collect::<Vec<_>>()
    }

    fn take_pending_event(&self) -> Option<QueuedEvent> {
        self.runtime.borrow_mut().pending_events.pop_front()
    }

    fn dispatch_observers(&self, source_id: u64) -> Result<(), RuntimeError> {
        let mut callbacks = {
            let mut runtime = self.runtime.borrow_mut();
            std::mem::take(runtime.observe_subscriptions.entry(source_id).or_default())
        };

        for subscription in &mut callbacks {
            if subscription.active.load(Ordering::Relaxed) {
                (subscription.callback)(&self.runtime, source_id)?;
            }
        }

        callbacks.retain(|subscription| subscription.active.load(Ordering::Relaxed));
        if !callbacks.is_empty() {
            self.runtime
                .borrow_mut()
                .observe_subscriptions
                .entry(source_id)
                .or_default()
                .append(&mut callbacks);
        }

        Ok(())
    }

    fn dispatch_event(&self, event: QueuedEvent) -> Result<(), RuntimeError> {
        let key = EventSubscriptionKey {
            source_id: event.source_id,
            event_type: event.event_type,
        };
        let mut callbacks = {
            let mut runtime = self.runtime.borrow_mut();
            std::mem::take(runtime.event_subscriptions.entry(key).or_default())
        };

        for subscription in &mut callbacks {
            if subscription.active.load(Ordering::Relaxed) {
                (subscription.callback)(&self.runtime, event.source_id, event.payload.as_ref())?;
            }
        }

        callbacks.retain(|subscription| subscription.active.load(Ordering::Relaxed));
        if !callbacks.is_empty() {
            self.runtime
                .borrow_mut()
                .event_subscriptions
                .entry(key)
                .or_default()
                .append(&mut callbacks);
        }

        Ok(())
    }
}

fn render_view_entity<T>(
    entity_id: u64,
    runtime: &Rc<RefCell<RuntimeState>>,
    window: &WindowInfo,
) -> Result<AnyElement, RuntimeError>
where
    T: Render + 'static,
{
    let boxed = runtime
        .borrow_mut()
        .entities
        .remove(&entity_id)
        .ok_or(RuntimeError::EntityNotFound(entity_id))?;

    let mut typed = boxed
        .downcast::<T>()
        .map_err(|_| RuntimeError::TypeMismatch(entity_id))?;

    let (background_executor, ui_executor) = {
        let runtime = runtime.borrow();
        (
            runtime.background_executor.clone(),
            runtime.ui_executor.clone(),
        )
    };

    let rendered = {
        let mut cx = Context::new(
            Entity::from_raw(entity_id),
            runtime.clone(),
            background_executor,
            ui_executor,
        );
        typed.render(window, &mut cx).into_any_element()
    };

    runtime
        .borrow_mut()
        .entities
        .insert(entity_id, typed as Box<dyn Any>);

    Ok(rendered)
}

pub(in crate::app) fn merge_lane(
    map: &mut HashMap<u64, DirtyLaneMask>,
    entity_id: u64,
    lane: DirtyLaneMask,
) {
    map.entry(entity_id)
        .and_modify(|existing| *existing |= lane)
        .or_insert(lane);
}
