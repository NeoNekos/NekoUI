use crate::style::{Display, ResolvedStyle};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OutputParticipation {
    layout: bool,
    paint: bool,
    hit_test: bool,
    semantics: bool,
}

impl OutputParticipation {
    pub const INCLUDED: Self = Self {
        layout: true,
        paint: true,
        hit_test: true,
        semantics: true,
    };

    pub const EXCLUDED: Self = Self {
        layout: false,
        paint: false,
        hit_test: false,
        semantics: false,
    };

    pub fn included() -> Self {
        Self::INCLUDED
    }

    pub fn excluded() -> Self {
        Self::EXCLUDED
    }

    pub fn layout(self) -> bool {
        self.layout
    }

    pub fn paint(self) -> bool {
        self.paint
    }

    pub fn hit_test(self) -> bool {
        self.hit_test
    }

    pub fn semantics(self) -> bool {
        self.semantics
    }

    pub(crate) fn resolve(parent: Self, display: Display) -> Self {
        if display == Display::None {
            Self::EXCLUDED
        } else {
            parent
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StyleNodeSnapshot {
    node_id: u64,
    node_generation: u64,
    resolved: ResolvedStyle,
    participation: OutputParticipation,
    children: Vec<StyleNodeSnapshot>,
}

impl StyleNodeSnapshot {
    pub(crate) fn new(
        node_id: u64,
        node_generation: u64,
        resolved: ResolvedStyle,
        participation: OutputParticipation,
        children: Vec<StyleNodeSnapshot>,
    ) -> Self {
        Self {
            node_id,
            node_generation,
            resolved,
            participation,
            children,
        }
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn node_generation(&self) -> u64 {
        self.node_generation
    }

    pub fn resolved(&self) -> &ResolvedStyle {
        &self.resolved
    }

    pub fn participation(&self) -> OutputParticipation {
        self.participation
    }

    pub fn children(&self) -> &[StyleNodeSnapshot] {
        &self.children
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StyleTreeSnapshot {
    node_count: usize,
    root: Option<StyleNodeSnapshot>,
}

impl StyleTreeSnapshot {
    pub(crate) fn new(node_count: usize, root: Option<StyleNodeSnapshot>) -> Self {
        Self { node_count, root }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn root(&self) -> Option<&StyleNodeSnapshot> {
        self.root.as_ref()
    }
}
