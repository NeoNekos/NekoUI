use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport, ProbeSnapshot};
use crate::error::NekoResult;
use crate::layout::LayoutTreeSnapshot;
use crate::retained::RetainedTreeSnapshot;
use crate::runtime::Runtime;
use crate::scene::PaintScene;
use crate::style::StyleTreeSnapshot;
use crate::window::{AnyWindowHandle, WindowRecord};

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

    pub fn retained_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<RetainedTreeSnapshot> {
        self.runtime.retained_snapshot(handle.into())
    }

    pub fn style_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<StyleTreeSnapshot> {
        self.runtime.style_snapshot(handle.into())
    }

    pub fn layout_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<LayoutTreeSnapshot> {
        self.runtime.layout_snapshot(handle.into())
    }

    pub fn scene_snapshot(&self, handle: impl Into<AnyWindowHandle>) -> NekoResult<PaintScene> {
        self.runtime.scene_snapshot(handle.into())
    }
}
