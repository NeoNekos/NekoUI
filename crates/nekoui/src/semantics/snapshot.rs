use crate::element::ElementKey;
use crate::layout::LayoutRect;
use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::semantics::SemanticBuildStats;
use crate::semantics::generation::SemanticGeneration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct SemanticNodeId {
    retained_id: RetainedNodeId,
    retained_generation: NodeGeneration,
}

impl SemanticNodeId {
    pub(crate) fn new(retained_id: RetainedNodeId, retained_generation: NodeGeneration) -> Self {
        Self {
            retained_id,
            retained_generation,
        }
    }

    #[cfg(test)]
    pub(crate) fn retained_id(self) -> RetainedNodeId {
        self.retained_id
    }

    #[cfg(test)]
    pub(crate) fn retained_generation(self) -> NodeGeneration {
        self.retained_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum SemanticRole {
    Window,
    Generic,
    Text,
    Textbox,
}

impl SemanticRole {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Generic => "generic",
            Self::Text => "text",
            Self::Textbox => "textbox",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum SemanticAction {
    Activate,
    Focus,
    Scroll,
    Edit,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SemanticStateSnapshot {
    focusable: bool,
    focused: bool,
    window_focused: bool,
    scrollable: bool,
    editable: bool,
    selection: Option<crate::interaction::TextRange>,
    composition: Option<crate::interaction::TextRange>,
    composition_cursor: Option<crate::interaction::TextRange>,
}

pub(crate) struct SemanticStateParts {
    pub(crate) focusable: bool,
    pub(crate) focused: bool,
    pub(crate) window_focused: bool,
    pub(crate) scrollable: bool,
    pub(crate) editable: bool,
    pub(crate) selection: Option<crate::interaction::TextRange>,
    pub(crate) composition: Option<crate::interaction::TextRange>,
    pub(crate) composition_cursor: Option<crate::interaction::TextRange>,
}

impl SemanticStateSnapshot {
    pub(crate) fn new(parts: SemanticStateParts) -> Self {
        Self {
            focusable: parts.focusable,
            focused: parts.focused,
            window_focused: parts.window_focused,
            scrollable: parts.scrollable,
            editable: parts.editable,
            selection: parts.selection,
            composition: parts.composition,
            composition_cursor: parts.composition_cursor,
        }
    }

    #[cfg(test)]
    pub(crate) fn focusable(&self) -> bool {
        self.focusable
    }

    #[cfg(test)]
    pub(crate) fn focused(&self) -> bool {
        self.focused
    }

    #[cfg(test)]
    pub(crate) fn window_focused(&self) -> bool {
        self.window_focused
    }

    #[cfg(test)]
    pub(crate) fn scrollable(&self) -> bool {
        self.scrollable
    }

    #[cfg(test)]
    pub(crate) fn editable(&self) -> bool {
        self.editable
    }

    #[cfg(test)]
    pub(crate) fn selection(&self) -> Option<crate::interaction::TextRange> {
        self.selection
    }

    #[cfg(test)]
    pub(crate) fn composition(&self) -> Option<crate::interaction::TextRange> {
        self.composition
    }

    #[cfg(test)]
    pub(crate) fn composition_cursor(&self) -> Option<crate::interaction::TextRange> {
        self.composition_cursor
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SemanticNodeSnapshot {
    id: SemanticNodeId,
    key: Option<ElementKey>,
    role: SemanticRole,
    name: Option<String>,
    value: Option<String>,
    bounds: LayoutRect,
    state: SemanticStateSnapshot,
    actions: Vec<SemanticAction>,
    children: Vec<SemanticNodeSnapshot>,
}

pub(crate) struct SemanticNodeParts {
    pub id: SemanticNodeId,
    pub key: Option<ElementKey>,
    pub role: SemanticRole,
    pub name: Option<String>,
    pub value: Option<String>,
    pub bounds: LayoutRect,
    pub state: SemanticStateSnapshot,
    pub actions: Vec<SemanticAction>,
    pub children: Vec<SemanticNodeSnapshot>,
}

impl SemanticNodeSnapshot {
    pub(crate) fn new(parts: SemanticNodeParts) -> Self {
        Self {
            id: parts.id,
            key: parts.key,
            role: parts.role,
            name: parts.name,
            value: parts.value,
            bounds: parts.bounds,
            state: parts.state,
            actions: parts.actions,
            children: parts.children,
        }
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> SemanticNodeId {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn key(&self) -> Option<&ElementKey> {
        self.key.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn role(&self) -> SemanticRole {
        self.role
    }

    #[cfg(test)]
    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn value(&self) -> Option<&str> {
        self.value.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn bounds(&self) -> LayoutRect {
        self.bounds
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> &SemanticStateSnapshot {
        &self.state
    }

    #[cfg(test)]
    pub(crate) fn actions(&self) -> &[SemanticAction] {
        &self.actions
    }

    #[cfg(test)]
    pub(crate) fn children(&self) -> &[SemanticNodeSnapshot] {
        &self.children
    }

    #[cfg(test)]
    pub(crate) fn find_by_key(&self, key: &str) -> Option<&SemanticNodeSnapshot> {
        if self
            .key
            .as_ref()
            .is_some_and(|current| current.as_str() == key)
        {
            return Some(self);
        }

        self.children
            .iter()
            .find_map(|child| child.find_by_key(key))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SemanticTreeSnapshot {
    generation: SemanticGeneration,
    node_count: usize,
    root: Option<SemanticNodeSnapshot>,
    stats: SemanticBuildStats,
}

impl SemanticTreeSnapshot {
    pub(crate) fn new(
        generation: SemanticGeneration,
        node_count: usize,
        root: Option<SemanticNodeSnapshot>,
        stats: SemanticBuildStats,
    ) -> Self {
        Self {
            generation,
            node_count,
            root,
            stats,
        }
    }

    pub(crate) fn generation(&self) -> &SemanticGeneration {
        &self.generation
    }

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.node_count
    }

    #[cfg(test)]
    pub(crate) fn root(&self) -> Option<&SemanticNodeSnapshot> {
        self.root.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> &SemanticBuildStats {
        &self.stats
    }

    #[cfg(test)]
    pub(crate) fn find_by_key(&self, key: &str) -> Option<&SemanticNodeSnapshot> {
        self.root.as_ref().and_then(|root| root.find_by_key(key))
    }
}
