use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbeSnapshot {
    diagnostics: DiagnosticSnapshot,
    performance: PerformanceReport,
}

impl ProbeSnapshot {
    pub fn new(diagnostics: DiagnosticSnapshot, performance: PerformanceReport) -> Self {
        Self {
            diagnostics,
            performance,
        }
    }

    pub fn diagnostics(&self) -> &DiagnosticSnapshot {
        &self.diagnostics
    }

    pub fn performance(&self) -> &PerformanceReport {
        &self.performance
    }
}
