use crate::style::{
    AlignItems, BackgroundFill, Color, CornerRadii, Direction, EdgeInsets, EdgeWidths,
    JustifyContent, Length, Size, Style,
};

use super::{AnyElement, Fragment, IntoElement, IntoElements, ParentElement};

#[derive(Debug, Clone, PartialEq)]
pub struct Div {
    pub(crate) key: Option<u64>,
    pub(crate) style: Style,
    pub(crate) children: Fragment,
}

pub fn div() -> Div {
    Div {
        key: None,
        style: Style::default(),
        children: Fragment::new(),
    }
}

impl Default for Div {
    fn default() -> Self {
        div()
    }
}

impl Div {
    pub fn key(mut self, key: u64) -> Self {
        self.key = Some(key);
        self
    }

    pub fn size(mut self, size: Size) -> Self {
        self.style.layout.size = size;
        self
    }

    pub fn width(mut self, width: Length) -> Self {
        self.style.layout.size.width = width;
        self
    }

    pub fn height(mut self, height: Length) -> Self {
        self.style.layout.size.height = height;
        self
    }

    pub fn padding(mut self, padding: EdgeInsets) -> Self {
        self.style.layout.padding = padding;
        self
    }

    pub fn margin(mut self, margin: EdgeInsets) -> Self {
        self.style.layout.margin = margin;
        self
    }

    pub fn direction(mut self, direction: Direction) -> Self {
        self.style.layout.direction = direction;
        self
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.style.layout.gap = gap;
        self
    }

    pub fn justify(mut self, justify_content: JustifyContent) -> Self {
        self.style.layout.justify_content = justify_content;
        self
    }

    pub fn align_items(mut self, align_items: AlignItems) -> Self {
        self.style.layout.align_items = align_items;
        self
    }

    pub fn bg(mut self, background: impl Into<BackgroundFill>) -> Self {
        self.style.paint.background = Some(background.into());
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.style.paint.corner_radii = CornerRadii::all(radius.max(0.0));
        self
    }

    pub fn corner_radii(mut self, corner_radii: CornerRadii) -> Self {
        self.style.paint.corner_radii = CornerRadii {
            top_left: corner_radii.top_left.max(0.0),
            top_right: corner_radii.top_right.max(0.0),
            bottom_right: corner_radii.bottom_right.max(0.0),
            bottom_left: corner_radii.bottom_left.max(0.0),
        };
        self
    }

    pub fn border(mut self, width: f32, color: Color) -> Self {
        self.style.paint.border_widths = EdgeWidths::all(width.max(0.0));
        self.style.paint.border_color = Some(color);
        self
    }

    pub fn border_widths(mut self, border_widths: EdgeWidths) -> Self {
        self.style.paint.border_widths = EdgeWidths {
            top: border_widths.top.max(0.0),
            right: border_widths.right.max(0.0),
            bottom: border_widths.bottom.max(0.0),
            left: border_widths.left.max(0.0),
        };
        self
    }

    pub fn border_color(mut self, color: Color) -> Self {
        self.style.paint.border_color = Some(color);
        self
    }

    pub fn opacity(mut self, opacity: f32) -> Self {
        self.style.paint.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn clip(mut self) -> Self {
        self.style.paint.clip_children = true;
        self
    }
}

impl ParentElement for Div {
    fn child(mut self, child: impl IntoElement) -> Self {
        self.children.push(child.into_any_element());
        self
    }

    fn children(mut self, children: impl IntoElements) -> Self {
        children.extend_into(&mut self.children);
        self
    }
}

impl IntoElement for Div {
    fn into_any_element(self) -> AnyElement {
        AnyElement::div(self)
    }
}
