use std::collections::BTreeMap;
use std::time::Duration;

use crate::diagnostic::DirtyLaneReport;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandIngressReport {
    pub commands_enqueued: u64,
    pub commands_processed: u64,
    pub queue_depth: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PerformanceReport {
    pub command_ingress: CommandIngressReport,
    pub notify_requests: u64,
    pub redraw_requests: u64,
    pub coalesced_redraws: u64,
    pub stale_handle_errors: u64,
    pub windows_alive: usize,
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
