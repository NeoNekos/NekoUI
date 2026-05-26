use std::marker::PhantomData;

use crate::app::Entity;
use crate::app::window_service::WindowService;
use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport};
use crate::element::IntoElement;
use crate::runtime::Runtime;
use crate::runtime::entity_store::EntityKey;

#[derive(Debug)]
pub struct AppContext<'a> {
    runtime: &'a mut Runtime,
}

impl<'a> AppContext<'a> {
    pub(crate) fn new(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
    }

    pub(crate) fn from_runtime(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
    }

    pub(crate) fn runtime(&mut self) -> &mut Runtime {
        self.runtime
    }

    pub fn new_entity<T: 'static>(
        &mut self,
        build: impl FnOnce(&mut Context<'_, T>) -> T,
    ) -> Entity<T> {
        let key = self.runtime.reserve_entity_key();
        let mut cx = Context::new(self.runtime, key);
        let value = build(&mut cx);
        self.runtime.insert_reserved_entity(key, value)
    }

    pub fn new_view<T: Render + 'static>(
        &mut self,
        build: impl FnOnce(&mut Context<'_, T>) -> T,
    ) -> Entity<T> {
        self.new_entity(build)
    }

    pub fn notify(&mut self) {
        self.runtime.request_notify();
    }

    pub fn windows(&mut self) -> WindowService<'_> {
        WindowService::new(self.runtime)
    }

    pub fn diagnostics(&self) -> DiagnosticSnapshot {
        self.runtime.diagnostics().snapshot()
    }

    pub fn performance_report(&self) -> PerformanceReport {
        self.runtime.performance_report()
    }
}

pub trait Render: 'static {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct Context<'a, T: 'static> {
    runtime: &'a mut Runtime,
    entity: EntityKey,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T: 'static> Context<'a, T> {
    pub(crate) fn new(runtime: &'a mut Runtime, entity: EntityKey) -> Self {
        Self {
            runtime,
            entity,
            marker: PhantomData,
        }
    }

    pub(crate) fn runtime(&mut self) -> &mut Runtime {
        self.runtime
    }

    pub(crate) fn entity_key(&self) -> EntityKey {
        self.entity
    }

    pub fn notify(&mut self) {
        self.runtime.notify_entity(self.entity);
    }

    pub fn windows(&mut self) -> WindowService<'_> {
        WindowService::new(self.runtime)
    }

    pub fn diagnostics(&self) -> DiagnosticSnapshot {
        self.runtime.diagnostics().snapshot()
    }

    pub fn performance_report(&self) -> PerformanceReport {
        self.runtime.performance_report()
    }
}
