use crate::layout::LayoutRect;
use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::scene::SceneInputSignature;
use crate::style::{Color, CornerRadii, Edges, Length, Opacity};
use crate::text::TextLayoutRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SceneOrder(u64);

impl SceneOrder {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    fill: Option<Color>,
    border_color: Option<Color>,
    border_width: Edges<Length>,
    corner_radius: CornerRadii<Length>,
    opacity: Opacity,
}

impl BoxShape {
    pub(crate) fn new(
        fill: Option<Color>,
        border_color: Option<Color>,
        border_width: Edges<Length>,
        corner_radius: CornerRadii<Length>,
        opacity: Opacity,
    ) -> Self {
        Self {
            fill,
            border_color,
            border_width,
            corner_radius,
            opacity,
        }
    }

    pub fn fill(self) -> Option<Color> {
        self.fill
    }

    pub fn border_color(self) -> Option<Color> {
        self.border_color
    }

    pub fn border_width(self) -> Edges<Length> {
        self.border_width
    }

    pub fn corner_radius(self) -> CornerRadii<Length> {
        self.corner_radius
    }

    pub fn opacity(self) -> Opacity {
        self.opacity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PaintFragmentKind {
    BoxShape {
        shape: BoxShape,
    },
    Rect {
        color: Color,
    },
    Text {
        text_generation: SceneInputSignature,
        text_metrics_generation: u64,
        color: Color,
    },
    ClipPush {
        clip: LayoutRect,
    },
    ClipPop,
    Unsupported {
        capability: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaintFragment {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
    order: SceneOrder,
    rect: LayoutRect,
    kind: PaintFragmentKind,
    text_layout: Option<TextLayoutRef>,
    clip: Option<LayoutRect>,
}

impl PaintFragment {
    pub(crate) fn new(
        node_id: RetainedNodeId,
        node_generation: NodeGeneration,
        order: SceneOrder,
        rect: LayoutRect,
        kind: PaintFragmentKind,
    ) -> Self {
        Self {
            node_id,
            node_generation,
            order,
            rect,
            kind,
            text_layout: None,
            clip: None,
        }
    }

    pub(crate) fn with_text_layout(mut self, layout: TextLayoutRef) -> Self {
        self.text_layout = Some(layout);
        self
    }

    pub(crate) fn with_clip(mut self, clip: LayoutRect) -> Self {
        self.clip = Some(clip);
        self
    }

    pub fn node_id(&self) -> RetainedNodeId {
        self.node_id
    }

    pub fn node_generation(&self) -> NodeGeneration {
        self.node_generation
    }

    pub fn order(&self) -> SceneOrder {
        self.order
    }

    pub fn rect(&self) -> LayoutRect {
        self.rect
    }

    pub fn kind(&self) -> &PaintFragmentKind {
        &self.kind
    }

    pub(crate) fn text_layout(&self) -> Option<&TextLayoutRef> {
        self.text_layout.as_ref()
    }

    pub(crate) fn clip(&self) -> Option<LayoutRect> {
        self.clip
    }
}
