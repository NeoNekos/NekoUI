use crate::diagnostic::DirtyLanes;
use crate::retained::RetainedIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirtyCause {
    WindowOpened,
    WindowClosed,
    AppNotified,
    RetainedChanged,
    NodeCreated,
    NodeDestroyed,
    NodeReplaced,
    NodeMoved,
    StyleChanged,
    TextChanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedDirty {
    pub identity: Option<RetainedIdentity>,
    pub cause: DirtyCause,
    pub lanes: DirtyLanes,
}

impl RetainedDirty {
    pub fn new(identity: Option<RetainedIdentity>, cause: DirtyCause, lanes: DirtyLanes) -> Self {
        Self {
            identity,
            cause,
            lanes,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RetainedDiffStats {
    pub old_node_count: usize,
    pub new_node_count: usize,
    pub preserved: usize,
    pub created: usize,
    pub moved_nodes: usize,
    pub replaced: usize,
    pub destroyed: usize,
    pub duplicate_keys: usize,
    pub kind_mismatches: usize,
}
