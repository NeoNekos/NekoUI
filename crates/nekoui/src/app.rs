use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use crossbeam_channel::{Receiver, Sender, bounded};
use hashbrown::{HashMap, HashSet};
use parking_lot::Mutex;

use crate::element::{Element, ElementKind, IntoElement, ViewSpec};
use crate::error::{Error, RuntimeError};
use crate::platform;
use crate::scene::DirtyLaneMask;
use crate::window::{Window, WindowHandle, WindowId, WindowOptions};

static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1);
pub struct Application {
    last_window_behavior: LastWindowBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastWindowBehavior {
    ExitEventLoop,
    KeepEventLoopAlive,
}

impl Application {
    pub fn new() -> Self {
        Self {
            last_window_behavior: LastWindowBehavior::ExitEventLoop,
        }
    }

    pub fn last_window_behavior(mut self, behavior: LastWindowBehavior) -> Self {
        self.last_window_behavior = behavior;
        self
    }

    pub fn run(
        self,
        on_launch: impl FnOnce(&mut App) -> Result<(), Error> + 'static,
    ) -> Result<(), Error> {
        platform::run_application(self.last_window_behavior, on_launch)
    }
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

type WakeHandle = Arc<dyn Fn() + Send + Sync>;
type WindowBuildFn = Box<dyn FnOnce(&mut Window, &mut App) -> Element + 'static>;
type ViewRenderer =
    fn(u64, &Rc<RefCell<RuntimeState>>, &mut Window) -> Result<Element, RuntimeError>;
type ObserveCallback = Box<dyn FnMut(&Rc<RefCell<RuntimeState>>, u64) -> Result<(), RuntimeError>>;
type EventCallback =
    Box<dyn FnMut(&Rc<RefCell<RuntimeState>>, u64, &dyn Any) -> Result<(), RuntimeError>>;

struct RuntimeState {
    background_executor: BackgroundExecutor,
    ui_executor: UiExecutor,
    entities: HashMap<u64, Box<dyn Any>>,
    dirty_entities: HashMap<u64, DirtyLaneMask>,
    view_renderers: HashMap<u64, ViewRenderer>,
    observe_subscriptions: HashMap<u64, Vec<ObserveSubscription>>,
    event_subscriptions: HashMap<EventSubscriptionKey, Vec<EventSubscription>>,
    pending_events: VecDeque<QueuedEvent>,
}

struct ObserveSubscription {
    active: Arc<AtomicBool>,
    callback: ObserveCallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EventSubscriptionKey {
    source_id: u64,
    event_type: TypeId,
}

struct EventSubscription {
    active: Arc<AtomicBool>,
    callback: EventCallback,
}

struct QueuedEvent {
    source_id: u64,
    event_type: TypeId,
    payload: Box<dyn Any>,
}

#[derive(Default)]
pub(crate) struct RuntimeProcessResult {
    pub dirty_views: HashMap<u64, DirtyLaneMask>,
}

pub(crate) struct PendingWindowRequest {
    pub id: WindowId,
    pub options: WindowOptions,
    pub build_root: WindowBuildFn,
}

pub struct App {
    runtime: Rc<RefCell<RuntimeState>>,
    pending_windows: VecDeque<PendingWindowRequest>,
    wake_handle: Option<WakeHandle>,
    background_executor: BackgroundExecutor,
    ui_executor: UiExecutor,
}

impl App {
    pub(crate) fn new() -> Self {
        let background_executor = BackgroundExecutor::new();
        let ui_executor = UiExecutor::new();
        Self {
            runtime: Rc::new(RefCell::new(RuntimeState {
                background_executor: background_executor.clone(),
                ui_executor: ui_executor.clone(),
                entities: HashMap::new(),
                dirty_entities: HashMap::new(),
                view_renderers: HashMap::new(),
                observe_subscriptions: HashMap::new(),
                event_subscriptions: HashMap::new(),
                pending_events: VecDeque::new(),
            })),
            pending_windows: VecDeque::new(),
            wake_handle: None,
            background_executor,
            ui_executor,
        }
    }

    pub(crate) fn set_wake_handle(&mut self, wake_handle: Option<WakeHandle>) {
        self.wake_handle = wake_handle.clone();
        self.ui_executor.set_wake_handle(wake_handle);
    }

    fn wake_runtime(&self) {
        if let Some(wake_handle) = &self.wake_handle {
            wake_handle();
        }
    }

    pub fn open_window<E>(
        &mut self,
        options: WindowOptions,
        build_root: impl FnOnce(&mut Window, &mut App) -> E + 'static,
    ) -> Result<WindowHandle, Error>
    where
        E: IntoElement<Element = Element>,
    {
        let id = WindowId::new();
        self.pending_windows.push_back(PendingWindowRequest {
            id,
            options,
            build_root: Box::new(move |window, app| build_root(window, app).into_element()),
        });
        self.wake_runtime();
        Ok(WindowHandle::new(id))
    }

    pub fn insert_entity<T>(&self, state: T) -> Entity<T>
    where
        T: 'static,
    {
        let entity = Entity::new();
        self.runtime
            .borrow_mut()
            .entities
            .insert(entity.id, Box::new(state));
        entity
    }

    pub fn insert_view<T>(&self, state: T) -> View<T>
    where
        T: Render + 'static,
    {
        let view = View::new();
        let mut runtime = self.runtime.borrow_mut();
        runtime.entities.insert(view.id, Box::new(state));
        runtime
            .view_renderers
            .insert(view.id, render_view_entity::<T>);
        view
    }

    pub fn update<T, R>(
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
            .remove(&entity.id)
            .ok_or(RuntimeError::EntityNotFound(entity.id))?;

        let mut typed = boxed
            .downcast::<T>()
            .map_err(|_| RuntimeError::TypeMismatch(entity.id))?;

        let result = {
            let mut cx = Context::new(
                entity,
                self.runtime.clone(),
                self.runtime.borrow().background_executor.clone(),
                self.runtime.borrow().ui_executor.clone(),
            );
            updater(typed.as_mut(), &mut cx)
        };

        self.runtime
            .borrow_mut()
            .entities
            .insert(entity.id, typed as Box<dyn Any>);

        Ok(result)
    }

    pub fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    pub fn ui_executor(&self) -> UiExecutor {
        self.ui_executor.clone()
    }

    pub(crate) fn drain_window_requests(&mut self) -> Vec<PendingWindowRequest> {
        self.pending_windows.drain(..).collect()
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

    pub(crate) fn resolve_root_element_with_views(
        &self,
        window: &mut Window,
        template: &Element,
    ) -> Result<(Element, HashSet<u64>), RuntimeError> {
        let mut referenced_views = HashSet::new();
        let resolved = self.resolve_element(window, template, &mut referenced_views)?;
        Ok((resolved, referenced_views))
    }

    pub(crate) fn make_runtime_window(
        id: WindowId,
        options: &WindowOptions,
        size: crate::window::WindowSize,
        physical_size: crate::window::WindowSize,
        scale_factor: f64,
    ) -> Window {
        Window::new_with_metrics(
            id,
            options.title_str().to_string(),
            size,
            physical_size,
            scale_factor,
        )
    }

    fn resolve_element(
        &self,
        window: &mut Window,
        template: &Element,
        referenced_views: &mut HashSet<u64>,
    ) -> Result<Element, RuntimeError> {
        match template.kind() {
            ElementKind::Div(div) => {
                let mut resolved = div.clone();
                resolved.children = div
                    .children
                    .iter()
                    .map(|child| self.resolve_element(window, child, referenced_views))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(resolved.into_element())
            }
            ElementKind::Text(text) => Ok(text.clone().into_element()),
            ElementKind::View(ViewSpec { entity_id }) => {
                referenced_views.insert(*entity_id);
                let renderer = self
                    .runtime
                    .borrow()
                    .view_renderers
                    .get(entity_id)
                    .copied()
                    .ok_or(RuntimeError::EntityNotFound(*entity_id))?;
                let rendered = renderer(*entity_id, &self.runtime, window)?;
                self.resolve_element(window, &rendered, referenced_views)
            }
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
        let mut callbacks = self
            .runtime
            .borrow_mut()
            .observe_subscriptions
            .remove(&source_id)
            .unwrap_or_default();

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
                .extend(callbacks);
        }

        Ok(())
    }

    fn dispatch_event(&self, event: QueuedEvent) -> Result<(), RuntimeError> {
        let key = EventSubscriptionKey {
            source_id: event.source_id,
            event_type: event.event_type,
        };
        let mut callbacks = self
            .runtime
            .borrow_mut()
            .event_subscriptions
            .remove(&key)
            .unwrap_or_default();

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
                .extend(callbacks);
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Entity<T> {
    id: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for Entity<T> {}

impl<T> Clone for Entity<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Entity<T> {
    fn new() -> Self {
        Self {
            id: NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed),
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn downgrade(self) -> WeakEntity<T> {
        WeakEntity {
            id: self.id,
            marker: PhantomData,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct WeakEntity<T> {
    id: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for WeakEntity<T> {}

impl<T> Clone for WeakEntity<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> WeakEntity<T> {
    pub fn id(&self) -> u64 {
        self.id
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct View<T> {
    id: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for View<T> {}

impl<T> Clone for View<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> View<T> {
    fn new() -> Self {
        Self {
            id: NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed),
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn entity(self) -> Entity<T> {
        Entity {
            id: self.id,
            marker: PhantomData,
        }
    }
}

impl<T> IntoElement for View<T> {
    type Element = Element;

    fn into_element(self) -> Self::Element {
        Element::new(ElementKind::View(ViewSpec { entity_id: self.id }))
    }
}

pub struct Context<'a, T> {
    entity: Entity<T>,
    runtime: Rc<RefCell<RuntimeState>>,
    background_executor: BackgroundExecutor,
    ui_executor: UiExecutor,
    marker: PhantomData<&'a mut T>,
}

pub trait EventEmitter<E> {}

impl<'a, T: 'static> Context<'a, T> {
    fn new(
        entity: Entity<T>,
        runtime: Rc<RefCell<RuntimeState>>,
        background_executor: BackgroundExecutor,
        ui_executor: UiExecutor,
    ) -> Self {
        Self {
            entity,
            runtime,
            background_executor,
            ui_executor,
            marker: PhantomData,
        }
    }

    pub fn entity(&self) -> Entity<T> {
        self.entity
    }

    pub fn observe<U>(
        &mut self,
        entity: &Entity<U>,
        f: impl FnMut(&mut T, Entity<U>, &mut Context<'_, T>) + 'static,
    ) -> Result<Subscription, RuntimeError>
    where
        U: 'static,
    {
        if !self.registration_target_exists(self.entity.id) {
            return Err(RuntimeError::EntityNotFound(self.entity.id));
        }
        if !self.runtime.borrow().entities.contains_key(&entity.id) {
            return Err(RuntimeError::EntityNotFound(entity.id));
        }

        let active = Arc::new(AtomicBool::new(true));
        let mut callback = f;
        let target_id = self.entity.id;
        let source_id = entity.id;

        self.runtime
            .borrow_mut()
            .observe_subscriptions
            .entry(source_id)
            .or_default()
            .push(ObserveSubscription {
                active: active.clone(),
                callback: Box::new(move |runtime, observed_id| {
                    let (background_executor, ui_executor) = {
                        let runtime = runtime.borrow();
                        (
                            runtime.background_executor.clone(),
                            runtime.ui_executor.clone(),
                        )
                    };
                    let boxed = runtime
                        .borrow_mut()
                        .entities
                        .remove(&target_id)
                        .ok_or(RuntimeError::EntityNotFound(target_id))?;
                    let mut typed = boxed
                        .downcast::<T>()
                        .map_err(|_| RuntimeError::TypeMismatch(target_id))?;

                    {
                        let mut cx = Context::new(
                            Entity {
                                id: target_id,
                                marker: PhantomData,
                            },
                            runtime.clone(),
                            background_executor,
                            ui_executor,
                        );
                        callback(
                            typed.as_mut(),
                            Entity {
                                id: observed_id,
                                marker: PhantomData,
                            },
                            &mut cx,
                        );
                    }

                    runtime
                        .borrow_mut()
                        .entities
                        .insert(target_id, typed as Box<dyn Any>);
                    Ok(())
                }),
            });

        Ok(Subscription { active })
    }

    pub fn subscribe<U, E>(
        &mut self,
        entity: &Entity<U>,
        f: impl FnMut(&mut T, Entity<U>, &E, &mut Context<'_, T>) + 'static,
    ) -> Result<Subscription, RuntimeError>
    where
        U: EventEmitter<E> + 'static,
        E: 'static,
    {
        if !self.registration_target_exists(self.entity.id) {
            return Err(RuntimeError::EntityNotFound(self.entity.id));
        }
        if !self.runtime.borrow().entities.contains_key(&entity.id) {
            return Err(RuntimeError::EntityNotFound(entity.id));
        }

        let active = Arc::new(AtomicBool::new(true));
        let mut callback = f;
        let target_id = self.entity.id;
        let source_id = entity.id;

        self.runtime
            .borrow_mut()
            .event_subscriptions
            .entry(EventSubscriptionKey {
                source_id,
                event_type: TypeId::of::<E>(),
            })
            .or_default()
            .push(EventSubscription {
                active: active.clone(),
                callback: Box::new(move |runtime, emitted_by, event| {
                    let (background_executor, ui_executor) = {
                        let runtime = runtime.borrow();
                        (
                            runtime.background_executor.clone(),
                            runtime.ui_executor.clone(),
                        )
                    };
                    let typed_event = event
                        .downcast_ref::<E>()
                        .ok_or(RuntimeError::EventTypeMismatch)?;
                    let boxed = runtime
                        .borrow_mut()
                        .entities
                        .remove(&target_id)
                        .ok_or(RuntimeError::EntityNotFound(target_id))?;
                    let mut typed = boxed
                        .downcast::<T>()
                        .map_err(|_| RuntimeError::TypeMismatch(target_id))?;

                    {
                        let mut cx = Context::new(
                            Entity {
                                id: target_id,
                                marker: PhantomData,
                            },
                            runtime.clone(),
                            background_executor,
                            ui_executor,
                        );
                        callback(
                            typed.as_mut(),
                            Entity {
                                id: emitted_by,
                                marker: PhantomData,
                            },
                            typed_event,
                            &mut cx,
                        );
                    }

                    runtime
                        .borrow_mut()
                        .entities
                        .insert(target_id, typed as Box<dyn Any>);
                    Ok(())
                }),
            });

        Ok(Subscription { active })
    }

    pub fn notify(&mut self) {
        self.invalidate(DirtyLaneMask::BUILD);
    }

    pub fn emit<E>(&mut self, event: E) -> Result<(), RuntimeError>
    where
        T: EventEmitter<E> + 'static,
        E: 'static,
    {
        self.runtime
            .borrow_mut()
            .pending_events
            .push_back(QueuedEvent {
                source_id: self.entity.id,
                event_type: TypeId::of::<E>(),
                payload: Box::new(event),
            });
        Ok(())
    }

    pub fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    pub fn ui_executor(&self) -> UiExecutor {
        self.ui_executor.clone()
    }

    pub(crate) fn invalidate(&mut self, lane: DirtyLaneMask) {
        merge_lane(
            &mut self.runtime.borrow_mut().dirty_entities,
            self.entity.id,
            lane.normalized(),
        );
    }

    fn registration_target_exists(&self, entity_id: u64) -> bool {
        entity_id == self.entity.id || self.runtime.borrow().entities.contains_key(&entity_id)
    }
}

pub struct Subscription {
    active: Arc<AtomicBool>,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Relaxed);
    }
}

pub trait Render: 'static + Sized {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) -> impl IntoElement<Element = Element>;
}

pub enum TaskResult<T> {
    Ready(T),
    Canceled,
}

pub struct Task<T> {
    receiver: Receiver<TaskResult<T>>,
    canceled: Arc<AtomicBool>,
}

impl<T> Task<T> {
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::Relaxed);
    }

    pub fn try_recv(&self) -> Option<TaskResult<T>> {
        self.receiver.try_recv().ok()
    }

    pub fn recv(self) -> TaskResult<T> {
        self.receiver.recv().unwrap_or(TaskResult::Canceled)
    }
}

#[derive(Clone)]
pub struct BackgroundExecutor {
    sender: Sender<Box<dyn FnOnce() + Send + 'static>>,
}

impl BackgroundExecutor {
    fn new() -> Self {
        let worker_count = thread::available_parallelism()
            .map(|value| value.get().min(4))
            .unwrap_or(2)
            .max(1);
        let (sender, receiver) = bounded::<Box<dyn FnOnce() + Send + 'static>>(256);

        for index in 0..worker_count {
            let worker_receiver = receiver.clone();
            thread::Builder::new()
                .name(format!("nekoui-bg-{index}"))
                .spawn(move || {
                    while let Ok(job) = worker_receiver.recv() {
                        job();
                    }
                })
                .expect("background executor worker must start");
        }

        Self { sender }
    }

    pub fn spawn<F>(&self, future: F) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawn_job(move || pollster::block_on(future))
    }

    pub fn spawn_blocking<F, R>(&self, f: F) -> Task<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.spawn_job(f)
    }

    fn spawn_job<F, R>(&self, job: F) -> Task<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (result_sender, result_receiver) = bounded(1);
        let canceled = Arc::new(AtomicBool::new(false));
        let canceled_for_job = canceled.clone();

        let enqueue_result = self.sender.try_send(Box::new(move || {
            if canceled_for_job.load(Ordering::Relaxed) {
                let _ = result_sender.send(TaskResult::Canceled);
                return;
            }

            let output = job();
            let result = if canceled_for_job.load(Ordering::Relaxed) {
                TaskResult::Canceled
            } else {
                TaskResult::Ready(output)
            };
            let _ = result_sender.send(result);
        }));

        if enqueue_result.is_err() {
            canceled.store(true, Ordering::Relaxed);
        }

        Task {
            receiver: result_receiver,
            canceled,
        }
    }
}

#[derive(Clone)]
pub struct UiExecutor {
    inner: Rc<UiExecutorInner>,
}

struct UiExecutorInner {
    queue: Mutex<VecDeque<Box<dyn FnOnce() + 'static>>>,
    wake_handle: Mutex<Option<WakeHandle>>,
}

impl UiExecutor {
    fn new() -> Self {
        Self {
            inner: Rc::new(UiExecutorInner {
                queue: Mutex::new(VecDeque::new()),
                wake_handle: Mutex::new(None),
            }),
        }
    }

    fn set_wake_handle(&self, wake_handle: Option<WakeHandle>) {
        *self.inner.wake_handle.lock() = wake_handle;
    }

    pub fn spawn<F>(&self, future: F) -> Task<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        let (result_sender, result_receiver) = bounded(1);
        let canceled = Arc::new(AtomicBool::new(false));
        let canceled_for_job = canceled.clone();

        self.inner.queue.lock().push_back(Box::new(move || {
            if canceled_for_job.load(Ordering::Relaxed) {
                let _ = result_sender.send(TaskResult::Canceled);
                return;
            }

            let output = pollster::block_on(future);
            let result = if canceled_for_job.load(Ordering::Relaxed) {
                TaskResult::Canceled
            } else {
                TaskResult::Ready(output)
            };
            let _ = result_sender.send(result);
        }));

        if let Some(wake_handle) = self.inner.wake_handle.lock().clone() {
            wake_handle();
        }

        Task {
            receiver: result_receiver,
            canceled,
        }
    }

    pub fn run_pending(&self) {
        let mut pending = VecDeque::new();
        std::mem::swap(&mut pending, &mut *self.inner.queue.lock());
        while let Some(job) = pending.pop_front() {
            job();
        }
    }
}

fn render_view_entity<T>(
    entity_id: u64,
    runtime: &Rc<RefCell<RuntimeState>>,
    window: &mut Window,
) -> Result<Element, RuntimeError>
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

    let rendered = {
        let mut cx = Context::new(
            Entity {
                id: entity_id,
                marker: PhantomData,
            },
            runtime.clone(),
            runtime.borrow().background_executor.clone(),
            runtime.borrow().ui_executor.clone(),
        );
        typed.render(window, &mut cx).into_element()
    };

    runtime
        .borrow_mut()
        .entities
        .insert(entity_id, typed as Box<dyn Any>);

    Ok(rendered)
}

fn merge_lane(map: &mut HashMap<u64, DirtyLaneMask>, entity_id: u64, lane: DirtyLaneMask) {
    map.entry(entity_id)
        .and_modify(|existing| *existing |= lane)
        .or_insert(lane);
}

#[cfg(test)]
mod tests {
    use super::{
        App, Application, Entity, EventEmitter, LastWindowBehavior, TaskResult, UiExecutor,
    };
    use crate::window::WindowOptions;

    #[derive(Default)]
    struct Counter {
        value: usize,
    }

    impl EventEmitter<usize> for Counter {}

    #[test]
    fn application_defaults_to_exit_when_last_window_closes() {
        let application = Application::new();
        assert_eq!(
            application.last_window_behavior,
            LastWindowBehavior::ExitEventLoop
        );
    }

    #[test]
    fn open_window_queues_request() {
        let mut app = App::new();
        let handle = app
            .open_window(WindowOptions::new().title("Neko"), |_window, _app| ())
            .unwrap();

        let requests = app.drain_window_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].id, handle.id());
    }

    #[test]
    fn update_round_trips_entity_state() {
        let app = App::new();
        let entity = app.insert_entity(String::from("neko"));

        let updated = app
            .update(entity, |value, cx| {
                value.push_str(" ui");
                cx.notify();
                value.clone()
            })
            .unwrap();

        assert_eq!(updated, "neko ui");
    }

    #[test]
    fn observe_marks_subscriber_view_dirty_after_source_notify() {
        let mut app = App::new();
        let source: Entity<Counter> = app.insert_entity(Counter::default());
        let observer = app.insert_entity(Counter::default());

        let _subscription = app
            .update(observer, |_state, cx| {
                cx.observe(&source, |observer_state, _, cx| {
                    observer_state.value += 1;
                    cx.notify();
                })
                .unwrap()
            })
            .unwrap();

        app.update(source, |state, cx| {
            state.value += 1;
            cx.notify();
        })
        .unwrap();

        let runtime = app.process_runtime().unwrap();
        assert!(runtime.dirty_views.is_empty());
        let observer_value = app.update(observer, |state, _| state.value).unwrap();
        assert_eq!(observer_value, 1);
    }

    #[test]
    fn subscribe_and_emit_deliver_typed_events() {
        let mut app = App::new();
        let source: Entity<Counter> = app.insert_entity(Counter::default());
        let observer = app.insert_entity(Counter::default());

        let _subscription = app
            .update(observer, |_state, cx| {
                cx.subscribe(&source, |observer_state, _, event: &usize, cx| {
                    observer_state.value += *event;
                    cx.notify();
                })
                .unwrap()
            })
            .unwrap();

        app.update(source, |_state, cx| cx.emit(7usize).unwrap())
            .unwrap();

        let runtime = app.process_runtime().unwrap();
        assert!(runtime.dirty_views.is_empty());
        let observer_value = app.update(observer, |state, _| state.value).unwrap();
        assert_eq!(observer_value, 7);
    }

    #[test]
    fn dropping_subscription_stops_event_delivery() {
        let mut app = App::new();
        let source: Entity<Counter> = app.insert_entity(Counter::default());
        let observer = app.insert_entity(Counter::default());

        let subscription = app
            .update(observer, |_state, cx| {
                cx.subscribe(&source, |observer_state, _, event: &usize, _| {
                    observer_state.value += *event;
                })
                .unwrap()
            })
            .unwrap();
        drop(subscription);

        app.update(source, |_state, cx| cx.emit(3usize).unwrap())
            .unwrap();
        app.process_runtime().unwrap();

        let observer_value = app.update(observer, |state, _| state.value).unwrap();
        assert_eq!(observer_value, 0);
    }

    #[test]
    fn ui_executor_runs_tasks_only_when_drained() {
        let executor = UiExecutor::new();
        let task = executor.spawn(async { 42usize });
        assert!(task.try_recv().is_none());
        executor.run_pending();
        matches!(task.recv(), TaskResult::Ready(42));
    }
}
