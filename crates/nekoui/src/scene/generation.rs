use crate::layout::LayoutGeneration;
use crate::retained::RetainedTreeGeneration;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneInputSignature {
    facts: Vec<SceneSignatureFact>,
}

impl SceneInputSignature {
    pub(crate) fn new(facts: Vec<SceneSignatureFact>) -> Self {
        Self { facts }
    }

    pub fn facts(&self) -> &[SceneSignatureFact] {
        &self.facts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SceneSignatureFact {
    Node {
        node_id: u64,
        node_generation: u64,
    },
    Participation {
        layout: bool,
        paint: bool,
        hit_test: bool,
        semantics: bool,
    },
    Display(u8),
    Background(Option<u64>),
    Opacity(u32),
    TextColor(u64),
    FontSize(u32),
    TextOverflow(u8),
    Overflow(u8),
    MaxLines(Option<usize>),
    TextPayload(Option<String>),
    ScrollOffset {
        target: u64,
        x: u32,
        y: u32,
    },
    TextInputFocus {
        target_id: Option<u64>,
        target_generation: Option<u64>,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneGeneration {
    retained: Option<RetainedTreeGeneration>,
    layout: Option<LayoutGeneration>,
    style: SceneInputSignature,
    viewport: u64,
    text: SceneInputSignature,
    scroll: SceneInputSignature,
}

impl SceneGeneration {
    pub(crate) fn new_with_scroll(
        retained: Option<RetainedTreeGeneration>,
        layout: Option<LayoutGeneration>,
        style: SceneInputSignature,
        viewport: u64,
        text: SceneInputSignature,
        scroll: SceneInputSignature,
    ) -> Self {
        Self {
            retained,
            layout,
            style,
            viewport,
            text,
            scroll,
        }
    }

    pub fn retained_generation(&self) -> Option<RetainedTreeGeneration> {
        self.retained
    }

    pub fn layout_generation(&self) -> Option<LayoutGeneration> {
        self.layout
    }

    pub fn viewport_generation(&self) -> u64 {
        self.viewport
    }

    pub fn style_generation(&self) -> &SceneInputSignature {
        &self.style
    }

    pub fn text_generation(&self) -> &SceneInputSignature {
        &self.text
    }

    pub fn scroll_generation(&self) -> &SceneInputSignature {
        &self.scroll
    }
}
