use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RetainedNodeId(u64);

impl RetainedNodeId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeGeneration(u64);

impl NodeGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RetainedTreeGeneration(u64);

impl RetainedTreeGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RetainedIdentity {
    id: RetainedNodeId,
    generation: NodeGeneration,
}

impl RetainedIdentity {
    pub fn new(id: RetainedNodeId, generation: NodeGeneration) -> Self {
        Self { id, generation }
    }

    pub fn id(self) -> RetainedNodeId {
        self.id
    }

    pub fn generation(self) -> NodeGeneration {
        self.generation
    }

    pub fn validate_against(self, current: RetainedIdentity) -> NekoResult<()> {
        if self == current {
            Ok(())
        } else {
            Err(NekoError::stale("retained identity generation mismatch"))
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IdentitySeed {
    next_id: u64,
    tree_generation: RetainedTreeGeneration,
}

impl Default for IdentitySeed {
    fn default() -> Self {
        Self {
            next_id: 1,
            tree_generation: RetainedTreeGeneration::INITIAL,
        }
    }
}

impl IdentitySeed {
    pub(crate) fn allocate(&mut self) -> RetainedIdentity {
        let identity =
            RetainedIdentity::new(RetainedNodeId::new(self.next_id), NodeGeneration::INITIAL);
        self.next_id += 1;
        identity
    }

    pub(crate) fn tree_generation(&self) -> RetainedTreeGeneration {
        self.tree_generation
    }

    pub(crate) fn mark_tree_changed(&mut self) -> RetainedTreeGeneration {
        self.tree_generation = self.tree_generation.next();
        self.tree_generation
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ErrorKind;
    use crate::retained::{NodeGeneration, RetainedIdentity, RetainedNodeId};

    use super::IdentitySeed;

    #[test]
    fn identity_seed_allocates_unique_stable_ids() {
        let mut seed = IdentitySeed::default();
        let first = seed.allocate();
        let second = seed.allocate();

        assert_ne!(first.id(), second.id());
        assert_eq!(first.generation(), NodeGeneration::INITIAL);
    }

    #[test]
    fn generation_mismatch_is_typed_stale() {
        let stale = RetainedIdentity::new(RetainedNodeId::new(1), NodeGeneration::INITIAL);
        let current = RetainedIdentity::new(RetainedNodeId::new(2), NodeGeneration::INITIAL);

        assert_eq!(
            stale.validate_against(current).unwrap_err().kind(),
            ErrorKind::Stale
        );
    }
}
