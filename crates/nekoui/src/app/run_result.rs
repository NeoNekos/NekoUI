use crate::diagnostic::{DiagnosticSnapshot, PerformanceReport, ProbeSnapshot};
use crate::error::NekoResult;
use crate::layout::LayoutTreeSnapshot;
use crate::retained::RetainedTreeSnapshot;
use crate::runtime::Runtime;
use crate::scene::PaintScene;
use crate::semantics::SemanticTreeSnapshot;
use crate::style::StyleTreeSnapshot;
use crate::window::{AnyWindowHandle, WindowRecord};

#[derive(Debug)]
pub(crate) struct TestRun {
    runtime: Runtime,
}

impl TestRun {
    pub(crate) fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }

    pub(crate) fn diagnostics(&self) -> DiagnosticSnapshot {
        self.runtime.diagnostics().snapshot()
    }

    pub(crate) fn performance_report(&self) -> PerformanceReport {
        self.runtime.performance_report()
    }

    pub(crate) fn probe_snapshot(&self) -> ProbeSnapshot {
        ProbeSnapshot::new(self.diagnostics(), self.performance_report())
    }

    pub(crate) fn windows(&self) -> Vec<&WindowRecord> {
        self.runtime.state().windows().collect()
    }

    pub(crate) fn retained_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<RetainedTreeSnapshot> {
        self.runtime.retained_snapshot(handle.into())
    }

    pub(crate) fn style_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<StyleTreeSnapshot> {
        self.runtime.style_snapshot(handle.into())
    }

    pub(crate) fn layout_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<LayoutTreeSnapshot> {
        self.runtime.layout_snapshot(handle.into())
    }

    pub(crate) fn semantic_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<SemanticTreeSnapshot> {
        self.runtime.semantic_snapshot(handle.into())
    }

    pub(crate) fn scene_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<PaintScene> {
        self.runtime.scene_snapshot(handle.into())
    }
}
