use crate::element::{ElementKey, ElementKind};
use crate::layout::{LayoutPoint, LayoutRect, LayoutSize, Viewport};
use crate::retained::{RetainedNodeId, RetainedTreeGeneration};
use crate::style::Overflow;
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
    scroll: ScrollGeometry,
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
            scroll: boxes.scroll,
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

    pub fn scroll(&self) -> ScrollGeometry {
        self.scroll
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
                scroll: self.scroll,
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
    pub scroll: ScrollGeometry,
    pub text_layout: Option<TextLayoutRef>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollGeometry {
    overflow: Overflow,
    viewport: LayoutRect,
    content_extent: LayoutSize,
}

impl ScrollGeometry {
    pub const fn new(overflow: Overflow, viewport: LayoutRect, content_extent: LayoutSize) -> Self {
        Self {
            overflow,
            viewport,
            content_extent,
        }
    }

    pub fn overflow(self) -> Overflow {
        self.overflow
    }

    pub fn viewport(self) -> LayoutRect {
        self.viewport
    }

    pub fn content_extent(self) -> LayoutSize {
        self.content_extent
    }

    pub fn scrollable(self) -> bool {
        self.overflow == Overflow::Scroll
            && (self.max_offset().x() > 0.0 || self.max_offset().y() > 0.0)
    }

    pub fn clips(self) -> bool {
        matches!(self.overflow, Overflow::Hidden | Overflow::Scroll)
    }

    pub fn max_offset(self) -> LayoutPoint {
        LayoutPoint::new(
            (self.content_extent.width() - self.viewport.width()).max(0.0),
            (self.content_extent.height() - self.viewport.height()).max(0.0),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GeometryKey {
    margin_rect: LayoutRect,
    border_rect: LayoutRect,
    padding_rect: LayoutRect,
    content_rect: LayoutRect,
    content_size: LayoutSize,
    scroll: ScrollGeometry,
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
