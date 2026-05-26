use crate::retained::{NodeGeneration, RetainedNodeId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InteractionTarget {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
}

impl InteractionTarget {
    pub(crate) fn new(node_id: RetainedNodeId, node_generation: NodeGeneration) -> Self {
        Self {
            node_id,
            node_generation,
        }
    }

    pub(crate) fn node_id(self) -> RetainedNodeId {
        self.node_id
    }

    pub(crate) fn node_generation(self) -> NodeGeneration {
        self.node_generation
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct InteractionState {
    hover: Option<InteractionTarget>,
    pressed: Option<InteractionTarget>,
}

impl InteractionState {
    pub(crate) fn hover(&self) -> Option<InteractionTarget> {
        self.hover
    }

    pub(crate) fn pressed(&self) -> Option<InteractionTarget> {
        self.pressed
    }

    pub(crate) fn set_hover(&mut self, target: Option<InteractionTarget>) {
        self.hover = target;
    }

    pub(crate) fn set_pressed(&mut self, target: Option<InteractionTarget>) {
        self.pressed = target;
    }
}
