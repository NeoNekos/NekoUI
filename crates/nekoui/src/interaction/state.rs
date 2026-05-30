use std::collections::BTreeMap;

use crate::interaction::Modifiers;
use crate::layout::LayoutPoint;
use crate::retained::{NodeGeneration, RetainedNodeId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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
    keyboard_focus: Option<InteractionTarget>,
    last_hover_position: Option<LayoutPoint>,
    scroll_offsets: BTreeMap<InteractionTarget, LayoutPoint>,
    modifiers: Modifiers,
    window_focused: bool,
}

impl InteractionState {
    pub(crate) fn hover(&self) -> Option<InteractionTarget> {
        self.hover
    }

    pub(crate) fn pressed(&self) -> Option<InteractionTarget> {
        self.pressed
    }

    pub(crate) fn keyboard_focus(&self) -> Option<InteractionTarget> {
        self.keyboard_focus
    }

    pub(crate) fn last_hover_position(&self) -> Option<LayoutPoint> {
        self.last_hover_position
    }

    pub(crate) fn scroll_offset(&self, target: InteractionTarget) -> LayoutPoint {
        self.scroll_offsets
            .get(&target)
            .copied()
            .unwrap_or(LayoutPoint::ZERO)
    }

    pub(crate) fn scroll_offsets(
        &self,
    ) -> impl Iterator<Item = (InteractionTarget, LayoutPoint)> + '_ {
        self.scroll_offsets
            .iter()
            .map(|(target, offset)| (*target, *offset))
    }

    #[cfg(test)]
    pub(crate) fn scroll_offsets_len(&self) -> usize {
        self.scroll_offsets.len()
    }

    #[cfg(test)]
    pub(crate) fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    pub(crate) fn window_focused(&self) -> bool {
        self.window_focused
    }

    pub(crate) fn set_hover(&mut self, target: Option<InteractionTarget>) {
        self.hover = target;
    }

    pub(crate) fn set_last_hover_position(&mut self, position: Option<LayoutPoint>) {
        self.last_hover_position = position;
    }

    pub(crate) fn set_pressed(&mut self, target: Option<InteractionTarget>) {
        self.pressed = target;
    }

    pub(crate) fn set_keyboard_focus(&mut self, target: Option<InteractionTarget>) {
        self.keyboard_focus = target;
    }

    pub(crate) fn set_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }

    pub(crate) fn set_window_focused(&mut self, focused: bool) {
        self.window_focused = focused;
    }

    pub(crate) fn set_scroll_offset(&mut self, target: InteractionTarget, offset: LayoutPoint) {
        if offset == LayoutPoint::ZERO {
            self.scroll_offsets.remove(&target);
        } else {
            self.scroll_offsets.insert(target, offset);
        }
    }

    pub(crate) fn retain_scroll_offsets(
        &mut self,
        mut keep: impl FnMut(InteractionTarget) -> bool,
    ) -> Vec<InteractionTarget> {
        let stale = self
            .scroll_offsets
            .keys()
            .copied()
            .filter(|target| !keep(*target))
            .collect::<Vec<_>>();
        for target in &stale {
            self.scroll_offsets.remove(target);
        }
        stale
    }
}
