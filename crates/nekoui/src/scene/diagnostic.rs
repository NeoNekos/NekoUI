#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResourceDemandKind {
    Glyph,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneResourceDemand {
    kind: ResourceDemandKind,
    owner_node_id: u64,
    expected_generation: SceneInputSignature,
}

impl SceneResourceDemand {
    pub(crate) fn new(
        kind: ResourceDemandKind,
        owner_node_id: u64,
        expected_generation: SceneInputSignature,
    ) -> Self {
        Self {
            kind,
            owner_node_id,
            expected_generation,
        }
    }

    pub fn kind(&self) -> ResourceDemandKind {
        self.kind
    }

    pub fn owner_node_id(&self) -> u64 {
        self.owner_node_id
    }

    pub fn expected_generation(&self) -> &SceneInputSignature {
        &self.expected_generation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneDiagnostic {
    message: &'static str,
    count: u64,
}

impl SceneDiagnostic {
    pub(crate) fn new(message: &'static str, count: u64) -> Self {
        Self { message, count }
    }

    pub fn message(&self) -> &'static str {
        self.message
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneCompileStats {
    pub node_count: usize,
    pub fragment_count: usize,
    pub hit_test_entry_count: usize,
    pub damage_region_count: usize,
    pub resource_demand_count: usize,
    pub unsupported_fragment_count: usize,
    pub stale_drop_count: u64,
    pub duration: std::time::Duration,
}
use crate::scene::SceneInputSignature;
