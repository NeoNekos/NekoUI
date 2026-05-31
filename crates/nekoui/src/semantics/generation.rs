use crate::element::ElementKind;
use crate::interaction::{InteractionState, InteractionTarget};
use crate::layout::LayoutGeneration;
use crate::retained::RetainedTreeGeneration;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SemanticInputSignature {
    facts: Vec<SemanticSignatureFact>,
}

impl SemanticInputSignature {
    pub(crate) fn new(facts: Vec<SemanticSignatureFact>) -> Self {
        Self { facts }
    }

    #[cfg(test)]
    pub(crate) fn facts(&self) -> &[SemanticSignatureFact] {
        &self.facts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SemanticSignatureFact {
    Node {
        node_id: u64,
        node_generation: u64,
        kind: u8,
    },
    Participation {
        semantics: bool,
    },
    Focusable(bool),
    Handlers {
        pointer: bool,
        click: bool,
        key: bool,
    },
    TextValue {
        len: usize,
        hash: u64,
    },
    EditableValue {
        len: usize,
        generation: u64,
        composing: bool,
    },
    WindowFocus(bool),
    KeyboardFocus {
        target_id: Option<u64>,
        target_generation: Option<u64>,
    },
    TextInputFocus {
        target_id: Option<u64>,
        target_generation: Option<u64>,
    },
    ScrollOffset {
        target: u64,
        x: u32,
        y: u32,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SemanticGeneration {
    retained: Option<RetainedTreeGeneration>,
    layout: Option<LayoutGeneration>,
    viewport: u64,
    style: SemanticInputSignature,
    semantic: SemanticInputSignature,
    interaction: SemanticInputSignature,
}

impl SemanticGeneration {
    pub(crate) fn new(
        retained: Option<RetainedTreeGeneration>,
        layout: Option<LayoutGeneration>,
        viewport: u64,
        style: SemanticInputSignature,
        semantic: SemanticInputSignature,
        interaction: SemanticInputSignature,
    ) -> Self {
        Self {
            retained,
            layout,
            viewport,
            style,
            semantic,
            interaction,
        }
    }

    #[cfg(test)]
    pub(crate) fn retained_generation(&self) -> Option<RetainedTreeGeneration> {
        self.retained
    }

    #[cfg(test)]
    pub(crate) fn layout_generation(&self) -> Option<LayoutGeneration> {
        self.layout
    }

    #[cfg(test)]
    pub(crate) fn viewport_generation(&self) -> u64 {
        self.viewport
    }

    #[cfg(test)]
    pub(crate) fn style_signature(&self) -> &SemanticInputSignature {
        &self.style
    }

    #[cfg(test)]
    pub(crate) fn semantic_signature(&self) -> &SemanticInputSignature {
        &self.semantic
    }

    #[cfg(test)]
    pub(crate) fn interaction_signature(&self) -> &SemanticInputSignature {
        &self.interaction
    }
}

pub(crate) fn element_kind_fact(kind: ElementKind) -> u8 {
    match kind {
        ElementKind::Div => 1,
        ElementKind::Text => 2,
        ElementKind::Input => 3,
    }
}

pub(crate) fn text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn scroll_target_signature(target: InteractionTarget) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in [target.node_id().raw(), target.node_generation().raw()] {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn interaction_signature(
    interaction: Option<&InteractionState>,
) -> SemanticInputSignature {
    let Some(interaction) = interaction else {
        return SemanticInputSignature::default();
    };
    let mut facts = Vec::new();
    facts.push(SemanticSignatureFact::WindowFocus(
        interaction.window_focused(),
    ));
    let keyboard_focus = interaction.keyboard_focus();
    facts.push(SemanticSignatureFact::KeyboardFocus {
        target_id: keyboard_focus.map(|target| target.node_id().raw()),
        target_generation: keyboard_focus.map(|target| target.node_generation().raw()),
    });
    let text_input_focus = interaction.text_input_focus();
    facts.push(SemanticSignatureFact::TextInputFocus {
        target_id: text_input_focus.map(|target| target.node_id().raw()),
        target_generation: text_input_focus.map(|target| target.node_generation().raw()),
    });
    facts.extend(interaction.scroll_offsets().map(|(target, offset)| {
        SemanticSignatureFact::ScrollOffset {
            target: scroll_target_signature(target),
            x: offset.x().to_bits(),
            y: offset.y().to_bits(),
        }
    }));
    SemanticInputSignature::new(facts)
}
