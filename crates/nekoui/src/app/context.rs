use crate::app::window_service::WindowService;
use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport};
use crate::runtime::Runtime;

#[derive(Debug)]
pub struct AppContext<'a> {
    runtime: &'a mut Runtime,
}

impl<'a> AppContext<'a> {
    pub(crate) fn new(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
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
