use std::borrow::Cow;

use crate::element::{ElementKey, ElementKind};
use crate::interaction::InteractionHandlers;
use crate::retained::{NodeGeneration, RetainedNodeId, RetainedTreeGeneration};
use crate::style::{OutputParticipation, ResolvedStyle, StyleDeclaration};
use crate::text::{EditableTextState, TextBlock};

#[derive(Clone, Debug, PartialEq)]
pub struct RetainedNodeSnapshot {
    pub(crate) id: RetainedNodeId,
    pub(crate) generation: NodeGeneration,
    pub(crate) kind: ElementKind,
    pub(crate) key: Option<ElementKey>,
    pub(crate) style: StyleDeclaration,
    pub(crate) focusable: bool,
    pub(crate) handlers: InteractionHandlers,
    pub(crate) resolved_style: ResolvedStyle,
    pub(crate) participation: OutputParticipation,
    pub(crate) text: Option<Cow<'static, str>>,
    pub(crate) editable: Option<EditableTextState>,
    pub(crate) children: Vec<RetainedNodeSnapshot>,
}

impl RetainedNodeSnapshot {
    pub fn id(&self) -> RetainedNodeId {
        self.id
    }

    pub fn generation(&self) -> NodeGeneration {
        self.generation
    }

    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    pub fn key(&self) -> Option<&ElementKey> {
        self.key.as_ref()
    }

    pub fn style(&self) -> &StyleDeclaration {
        &self.style
    }

    pub(crate) fn focusable(&self) -> bool {
        self.focusable
    }

    pub(crate) fn handlers(&self) -> &InteractionHandlers {
        &self.handlers
    }

    pub fn resolved_style(&self) -> &ResolvedStyle {
        &self.resolved_style
    }

    pub fn participation(&self) -> OutputParticipation {
        self.participation
    }

    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub(crate) fn editable(&self) -> Option<&EditableTextState> {
        self.editable.as_ref()
    }

    pub(crate) fn text_block(&self) -> Option<&TextBlock> {
        self.editable.as_ref().map(EditableTextState::block)
    }

    pub(crate) fn display_text(&self) -> Option<String> {
        self.editable
            .as_ref()
            .map(|editable| editable.block().display_text())
            .or_else(|| self.text().map(ToOwned::to_owned))
    }

    pub(crate) fn has_display_text(&self) -> bool {
        self.editable.is_some() || self.text.is_some()
    }

    pub fn children(&self) -> &[RetainedNodeSnapshot] {
        &self.children
    }

    pub fn find_by_key(&self, key: &str) -> Option<&RetainedNodeSnapshot> {
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
pub struct RetainedTreeSnapshot {
    generation: Option<RetainedTreeGeneration>,
    node_count: usize,
    root: Option<RetainedNodeSnapshot>,
}

impl RetainedTreeSnapshot {
    pub(crate) fn new(
        generation: Option<RetainedTreeGeneration>,
        node_count: usize,
        root: Option<RetainedNodeSnapshot>,
    ) -> Self {
        Self {
            generation,
            node_count,
            root,
        }
    }

    pub fn generation(&self) -> Option<RetainedTreeGeneration> {
        self.generation
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn root(&self) -> Option<&RetainedNodeSnapshot> {
        self.root.as_ref()
    }

    pub fn find_by_key(&self, key: &str) -> Option<&RetainedNodeSnapshot> {
        self.root.as_ref().and_then(|root| root.find_by_key(key))
    }
}
