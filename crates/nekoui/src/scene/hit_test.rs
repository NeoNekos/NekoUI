use crate::layout::{LayoutPoint, LayoutRect};
use crate::retained::{NodeGeneration, RetainedNodeId};

use super::fragment::SceneOrder;

#[derive(Clone, Debug, PartialEq)]
pub struct HitTestEntry {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
    rect: LayoutRect,
    order: SceneOrder,
}

impl HitTestEntry {
    pub(crate) fn new(
        node_id: RetainedNodeId,
        node_generation: NodeGeneration,
        rect: LayoutRect,
        order: SceneOrder,
    ) -> Self {
        Self {
            node_id,
            node_generation,
            rect,
            order,
        }
    }

    pub fn node_id(&self) -> RetainedNodeId {
        self.node_id
    }

    pub fn node_generation(&self) -> NodeGeneration {
        self.node_generation
    }

    pub fn rect(&self) -> LayoutRect {
        self.rect
    }

    pub fn order(&self) -> SceneOrder {
        self.order
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HitTestScene {
    entries: Vec<HitTestEntry>,
}

impl HitTestScene {
    pub(crate) fn new(entries: Vec<HitTestEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[HitTestEntry] {
        &self.entries
    }

    pub fn hit_test(&self, position: LayoutPoint) -> Option<&HitTestEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.rect().contains(position))
            .max_by_key(|entry| entry.order())
    }
}
