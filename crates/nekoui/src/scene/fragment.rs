use crate::layout::LayoutRect;
use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::scene::SceneInputSignature;
use crate::style::Color;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SceneOrder(u64);

impl SceneOrder {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PaintFragmentKind {
    Rect {
        color: Color,
    },
    Text {
        text_generation: SceneInputSignature,
        text_metrics_generation: u64,
        color: Color,
    },
    ClipPush {
        clip: LayoutRect,
    },
    ClipPop,
    Unsupported {
        capability: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaintFragment {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
    order: SceneOrder,
    rect: LayoutRect,
    kind: PaintFragmentKind,
}

impl PaintFragment {
    pub(crate) fn new(
        node_id: RetainedNodeId,
        node_generation: NodeGeneration,
        order: SceneOrder,
        rect: LayoutRect,
        kind: PaintFragmentKind,
    ) -> Self {
        Self {
            node_id,
            node_generation,
            order,
            rect,
            kind,
        }
    }

    pub fn node_id(&self) -> RetainedNodeId {
        self.node_id
    }

    pub fn node_generation(&self) -> NodeGeneration {
        self.node_generation
    }

    pub fn order(&self) -> SceneOrder {
        self.order
    }

    pub fn rect(&self) -> LayoutRect {
        self.rect
    }

    pub fn kind(&self) -> &PaintFragmentKind {
        &self.kind
    }
}
