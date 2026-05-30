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
}

impl SemanticRole {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Generic => "generic",
            Self::Text => "text",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum SemanticAction {
    Activate,
    Focus,
    Scroll,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SemanticStateSnapshot {
    focusable: bool,
    focused: bool,
    window_focused: bool,
    scrollable: bool,
}

impl SemanticStateSnapshot {
    pub(crate) fn new(
        focusable: bool,
        focused: bool,
        window_focused: bool,
        scrollable: bool,
    ) -> Self {
        Self {
            focusable,
            focused,
            window_focused,
            scrollable,
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
