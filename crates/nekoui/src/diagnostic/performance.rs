use std::collections::BTreeMap;
use std::time::Duration;

use crate::diagnostic::DirtyLaneReport;
use crate::retained::RetainedDiffStats;
use crate::scene::SceneCompileStats;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandIngressReport {
    pub commands_enqueued: u64,
    pub commands_processed: u64,
    pub queue_depth: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RetainedPerformanceReport {
    pub node_count: usize,
    pub diff_count: u64,
    pub last_diff: RetainedDiffStats,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StylePerformanceReport {
    pub resolved_node_count: usize,
    pub resolve_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayoutPassReport {
    pub node_count: usize,
    pub changed_geometry_count: usize,
    pub text_query_count: u64,
    pub text_cache_hits: u64,
    pub text_cache_misses: u64,
    pub blocked_on_text_count: u64,
    pub deferred_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayoutPerformanceReport {
    pub node_count: usize,
    pub pass_count: u64,
    pub text_query_count: u64,
    pub text_cache_hits: u64,
    pub text_cache_misses: u64,
    pub blocked_on_text_count: u64,
    pub deferred_count: u64,
    pub last_pass: LayoutPassReport,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextPerformanceReport {
    pub measure_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub deferred_count: u64,
    pub failed_count: u64,
    pub total_duration: Duration,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScenePerformanceReport {
    pub compile_count: u64,
    pub published_node_count: usize,
    pub last_compile: SceneCompileStats,
    pub node_count: usize,
    pub fragment_count: usize,
    pub hit_test_entry_count: usize,
    pub damage_region_count: usize,
    pub resource_demand_count: usize,
    pub stale_drop_count: u64,
    pub unsupported_fragment_count: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderFrameGraphReport {
    pub surface_generation: Option<u64>,
    pub pass_count: usize,
    pub draw_item_count: usize,
    pub upload_intent_count: usize,
    pub layer_count: usize,
    pub unsupported_fragment_count: usize,
    pub stale_drop_count: u64,
    pub duration: Duration,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderPerformanceReport {
    pub frame_graph_count: u64,
    pub pass_count: u64,
    pub upload_plan_count: u64,
    pub layer_count: u64,
    pub stale_drop_count: u64,
    pub unsupported_count: u64,
    pub prepared_frame_count: usize,
    pub last_frame_graph: RenderFrameGraphReport,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuPerformanceReport {
    pub backend_selected_count: u64,
    pub surface_state_count: u64,
    pub frame_phase_count: u64,
    pub presented_count: u64,
    pub not_renderable_count: u64,
    pub stale_drop_count: u64,
    pub unsupported_count: u64,
    pub recovery_count: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PerformanceReport {
    pub command_ingress: CommandIngressReport,
    pub notify_requests: u64,
    pub redraw_requests: u64,
    pub coalesced_redraws: u64,
    pub stale_handle_errors: u64,
    pub windows_alive: usize,
    pub retained: RetainedPerformanceReport,
    pub style: StylePerformanceReport,
    pub layout: LayoutPerformanceReport,
    pub text: TextPerformanceReport,
    pub scene: ScenePerformanceReport,
    pub render: RenderPerformanceReport,
    pub gpu: GpuPerformanceReport,
    pub dirty_lanes: Vec<DirtyLaneReport>,
    pub phase_durations: BTreeMap<&'static str, Duration>,
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::PerformanceReport;

    #[test]
    fn performance_report_has_probe_fields_not_fps() {
        let report = PerformanceReport::default();

        assert_eq!(report.command_ingress.commands_enqueued, 0);
        assert_eq!(report.coalesced_redraws, 0);
    }
}
