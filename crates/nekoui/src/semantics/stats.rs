use std::time::Duration;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SemanticBuildStats {
    pub node_count: usize,
    pub diagnostic_count: usize,
    pub stale_drop_count: u64,
    pub duration: Duration,
}
