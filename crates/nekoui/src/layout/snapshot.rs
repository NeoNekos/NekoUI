use crate::element::{ElementKey, ElementKind};
use crate::layout::{LayoutRect, LayoutSize, Viewport};
use crate::retained::{RetainedNodeId, RetainedTreeGeneration};
use crate::text::TextLayoutRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LayoutGeneration(u64);

impl LayoutGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn raw(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutNodeSnapshot {
    node_id: RetainedNodeId,
    kind: ElementKind,
    key: Option<ElementKey>,
    margin_rect: LayoutRect,
    border_rect: LayoutRect,
    padding_rect: LayoutRect,
    content_rect: LayoutRect,
    content_size: LayoutSize,
    text_layout: Option<TextLayoutRef>,
    children: Vec<LayoutNodeSnapshot>,
}

impl LayoutNodeSnapshot {
    pub(crate) fn new(
        node_id: RetainedNodeId,
        kind: ElementKind,
        key: Option<ElementKey>,
        boxes: LayoutBoxes,
        children: Vec<LayoutNodeSnapshot>,
    ) -> Self {
        Self {
            node_id,
            kind,
            key,
            margin_rect: boxes.margin_rect,
            border_rect: boxes.border_rect,
            padding_rect: boxes.padding_rect,
            content_rect: boxes.content_rect,
            content_size: boxes.content_size,
            text_layout: boxes.text_layout,
            children,
        }
    }

    pub fn node_id(&self) -> RetainedNodeId {
        self.node_id
    }

    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    pub fn key(&self) -> Option<&ElementKey> {
        self.key.as_ref()
    }

    pub fn margin_rect(&self) -> LayoutRect {
        self.margin_rect
    }

    pub fn border_rect(&self) -> LayoutRect {
        self.border_rect
    }

    pub fn padding_rect(&self) -> LayoutRect {
        self.padding_rect
    }

    pub fn content_rect(&self) -> LayoutRect {
        self.content_rect
    }

    pub fn content_size(&self) -> LayoutSize {
        self.content_size
    }

    pub(crate) fn text_layout(&self) -> Option<&TextLayoutRef> {
        self.text_layout.as_ref()
    }

    pub fn children(&self) -> &[LayoutNodeSnapshot] {
        &self.children
    }

    pub fn find_by_key(&self, key: &str) -> Option<&LayoutNodeSnapshot> {
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

    pub(crate) fn collect_geometry(&self, out: &mut Vec<(RetainedNodeId, GeometryKey)>) {
        out.push((
            self.node_id,
            GeometryKey {
                margin_rect: self.margin_rect,
                border_rect: self.border_rect,
                padding_rect: self.padding_rect,
                content_rect: self.content_rect,
                content_size: self.content_size,
            },
        ));
        for child in &self.children {
            child.collect_geometry(out);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutBoxes {
    pub margin_rect: LayoutRect,
    pub border_rect: LayoutRect,
    pub padding_rect: LayoutRect,
    pub content_rect: LayoutRect,
    pub content_size: LayoutSize,
    pub text_layout: Option<TextLayoutRef>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GeometryKey {
    margin_rect: LayoutRect,
    border_rect: LayoutRect,
    padding_rect: LayoutRect,
    content_rect: LayoutRect,
    content_size: LayoutSize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutTreeSnapshot {
    generation: Option<LayoutGeneration>,
    retained_generation: Option<RetainedTreeGeneration>,
    viewport: Viewport,
    node_count: usize,
    root: Option<LayoutNodeSnapshot>,
}

impl LayoutTreeSnapshot {
    pub(crate) fn new(
        generation: Option<LayoutGeneration>,
        retained_generation: Option<RetainedTreeGeneration>,
        viewport: Viewport,
        node_count: usize,
        root: Option<LayoutNodeSnapshot>,
    ) -> Self {
        Self {
            generation,
            retained_generation,
            viewport,
            node_count,
            root,
        }
    }

    pub fn generation(&self) -> Option<LayoutGeneration> {
        self.generation
    }

    pub fn retained_generation(&self) -> Option<RetainedTreeGeneration> {
        self.retained_generation
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn root(&self) -> Option<&LayoutNodeSnapshot> {
        self.root.as_ref()
    }

    pub fn find_by_key(&self, key: &str) -> Option<&LayoutNodeSnapshot> {
        self.root.as_ref().and_then(|root| root.find_by_key(key))
    }

    pub(crate) fn geometry(&self) -> Vec<(RetainedNodeId, GeometryKey)> {
        let mut geometry = Vec::with_capacity(self.node_count);
        if let Some(root) = &self.root {
            root.collect_geometry(&mut geometry);
        }
        geometry
    }
}

impl Default for LayoutTreeSnapshot {
    fn default() -> Self {
        Self::new(None, None, Viewport::default(), 0, None)
    }
}
