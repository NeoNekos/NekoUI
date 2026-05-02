use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, RuntimeError};
use crate::input::TextInputEvent;
use crate::scene::DirtyLaneMask;
use crate::window::{DisplayInfo, WindowHandle, WindowInfo, WindowOptions};

use super::executor::{BackgroundExecutor, UiExecutor};
use super::handle::{Entity, View};
use super::runtime::{
    EventSubscription, EventSubscriptionKey, ObserveSubscription, PendingWindowRequest,
    QueuedEvent, RuntimeState, TextInputSubscription, merge_lane,
};
use super::{App, Render};

type ImmediateWindowOpener<'a> = dyn FnMut(PendingWindowRequest) -> Result<(), Error> + 'a;

pub struct AppContext<'a> {
    app: NonNull<App>,
    immediate_window_opener: Option<NonNull<ImmediateWindowOpener<'a>>>,
    marker: PhantomData<&'a mut App>,
}

impl<'a> AppContext<'a> {
    pub(crate) fn new(app: &'a mut App) -> Self {
        Self {
            app: NonNull::from(app),
            immediate_window_opener: None,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_immediate_window_opener(
        app: &'a mut App,
        immediate_window_opener: &'a mut ImmediateWindowOpener<'a>,
    ) -> Self {
        Self {
            app: NonNull::from(app),
            immediate_window_opener: Some(NonNull::from(immediate_window_opener)),
            marker: PhantomData,
        }
    }

    pub fn open_window<V>(
        &mut self,
        options: WindowOptions,
        build_root: impl FnOnce(&WindowInfo, &mut AppContext<'_>) -> View<V> + 'static,
    ) -> Result<WindowHandle<V>, Error>
    where
        V: Render + 'static,
    {
        if let Some(immediate_window_opener) = self.immediate_window_opener {
            let (request, handle) = self.app_mut().prepare_window_request(options, build_root);
            // SAFETY: `immediate_window_opener` is created from a live `&mut` callback owned by
            // the runtime and only accessed through this `&mut AppContext` on the UI thread.
            unsafe { immediate_window_opener.as_ptr().as_mut() }
                .expect("immediate window opener pointer must be valid")(request)?;
            Ok(handle)
        } else {
            self.app_mut().open_window(options, build_root)
        }
    }

    pub fn new_view<T>(&mut self, init: impl FnOnce(&mut AppContext<'_>) -> T) -> View<T>
    where
        T: Render + 'static,
    {
        let state = init(self);
        self.app_mut().insert_view(state)
    }

    pub fn new_entity<T>(&mut self, init: impl FnOnce(&mut AppContext<'_>) -> T) -> Entity<T>
    where
        T: 'static,
    {
        let state = init(self);
        self.app_mut().insert_entity(state)
    }

    pub fn update<T, R>(
        &mut self,
        entity: Entity<T>,
        updater: impl FnOnce(&mut T, &mut Context<'_, T>) -> R,
    ) -> Result<R, RuntimeError>
    where
        T: 'static,
    {
        self.app_mut().update(entity, updater)
    }

    pub fn background_executor(&self) -> BackgroundExecutor {
        self.app_ref().background_executor()
    }

    pub fn ui_executor(&self) -> UiExecutor {
        self.app_ref().ui_executor()
    }

    pub fn displays(&self) -> Vec<DisplayInfo> {
        self.app_ref().displays()
    }

    pub fn primary_display(&self) -> Option<DisplayInfo> {
        self.app_ref().primary_display()
    }

    pub fn active_display(&self) -> Option<DisplayInfo> {
        self.app_ref().active_display()
    }

    fn app_ref(&self) -> &App {
        // SAFETY: `AppContext` is only constructed from a live `&mut App` owned by the UI domain.
        unsafe { self.app.as_ref() }
    }

    fn app_mut(&mut self) -> &mut App {
        // SAFETY: `AppContext` is only used on the UI domain and carries exclusive access via
        // `&mut self`. Nested framework-created `AppContext` values are serialized by the runtime.
        unsafe { self.app.as_mut() }
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
    pub(super) fn new(
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
        if !self.registration_target_exists(self.entity.id()) {
            return Err(RuntimeError::EntityNotFound(self.entity.id()));
        }
        if !self.runtime.borrow().entities.contains_key(&entity.id()) {
            return Err(RuntimeError::EntityNotFound(entity.id()));
        }

        let active = Arc::new(AtomicBool::new(true));
        let mut callback = f;
        let target_id = self.entity.id();
        let source_id = entity.id();

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
                            Entity::from_raw(target_id),
                            runtime.clone(),
                            background_executor,
                            ui_executor,
                        );
                        callback(typed.as_mut(), Entity::from_raw(observed_id), &mut cx);
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
        if !self.registration_target_exists(self.entity.id()) {
            return Err(RuntimeError::EntityNotFound(self.entity.id()));
        }
        if !self.runtime.borrow().entities.contains_key(&entity.id()) {
            return Err(RuntimeError::EntityNotFound(entity.id()));
        }

        let active = Arc::new(AtomicBool::new(true));
        let mut callback = f;
        let target_id = self.entity.id();
        let source_id = entity.id();

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
                            Entity::from_raw(target_id),
                            runtime.clone(),
                            background_executor,
                            ui_executor,
                        );
                        callback(
                            typed.as_mut(),
                            Entity::from_raw(emitted_by),
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
                source_id: self.entity.id(),
                event_type: TypeId::of::<E>(),
                payload: Box::new(event),
            });
        Ok(())
    }

    pub fn subscribe_text_input<U>(
        &mut self,
        entity: &Entity<U>,
        f: impl FnMut(&mut T, Entity<U>, &TextInputEvent, &mut Context<'_, T>) + 'static,
    ) -> Result<Subscription, RuntimeError>
    where
        U: 'static,
    {
        if !self.registration_target_exists(self.entity.id()) {
            return Err(RuntimeError::EntityNotFound(self.entity.id()));
        }
        let active = Arc::new(AtomicBool::new(true));
        let mut callback = f;
        let target_id = self.entity.id();
        let source_id = entity.id();

        self.runtime
            .borrow_mut()
            .text_input_subscriptions
            .entry(source_id)
            .or_default()
            .push(TextInputSubscription {
                active: active.clone(),
                callback: Box::new(move |runtime, emitted_by, event| {
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
                            Entity::from_raw(target_id),
                            runtime.clone(),
                            background_executor,
                            ui_executor,
                        );
                        callback(
                            typed.as_mut(),
                            Entity::from_raw(emitted_by),
                            event,
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

    #[cfg(test)]
    pub(crate) fn subscribe_text_input_target<U>(
        &mut self,
        entity: &Entity<U>,
        target: crate::input::InputNodeId,
        mut f: impl FnMut(&mut T, Entity<U>, &TextInputEvent, &mut Context<'_, T>) + 'static,
    ) -> Result<Subscription, RuntimeError>
    where
        U: 'static,
    {
        self.subscribe_text_input(entity, move |state, source, event, cx| {
            let target_matches = match event {
                TextInputEvent::Commit { target: event_target, .. }
                | TextInputEvent::Preedit { target: event_target, .. } => *event_target == target,
            };
            if target_matches {
                f(state, source, event, cx);
            }
        })
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
            self.entity.id(),
            lane.normalized(),
        );
    }

    fn registration_target_exists(&self, entity_id: u64) -> bool {
        entity_id == self.entity.id() || self.runtime.borrow().entities.contains_key(&entity_id)
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
