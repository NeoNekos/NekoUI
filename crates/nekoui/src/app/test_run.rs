use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport, ProbeSnapshot};
use crate::runtime::Runtime;
use crate::window::WindowRecord;

#[derive(Debug)]
pub struct TestRun {
    runtime: Runtime,
}

impl TestRun {
    pub(crate) fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }

    pub fn diagnostics(&self) -> DiagnosticSnapshot {
        self.runtime.diagnostics().snapshot()
    }

    pub fn performance_report(&self) -> PerformanceReport {
        self.runtime.performance_report()
    }

    pub fn probe_snapshot(&self) -> ProbeSnapshot {
        ProbeSnapshot::new(self.diagnostics(), self.performance_report())
    }

    pub fn windows(&self) -> Vec<&WindowRecord> {
        self.runtime.state().windows().collect()
    }
}
