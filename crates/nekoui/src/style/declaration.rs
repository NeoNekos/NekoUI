use crate::diagnostic::{DirtyLane, DirtyLanes};
use crate::error::NekoResult;
use crate::style::{Color, CornerRadii, Dimension, Edges, Length, Opacity};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Display {
    None,
    #[default]
    Block,
    Flex,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Overflow {
    #[default]
    Visible,
    Hidden,
    Scroll,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TextOverflow {
    Visible,
    #[default]
    Clip,
    Ellipsis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutStyleDeclaration {
    display: Option<Display>,
    width: Option<Dimension>,
    height: Option<Dimension>,
    padding: Edges<Option<Length>>,
    margin: Edges<Option<Length>>,
    border_width: Edges<Option<Length>>,
    gap: Option<Length>,
    overflow: Option<Overflow>,
}

impl Default for LayoutStyleDeclaration {
    fn default() -> Self {
        Self {
            display: None,
            width: None,
            height: None,
            padding: Edges::all(None),
            margin: Edges::all(None),
            border_width: Edges::all(None),
            gap: None,
            overflow: None,
        }
    }
}

impl LayoutStyleDeclaration {
    pub fn display(&self) -> Option<Display> {
        self.display
    }

    pub fn width(&self) -> Option<Dimension> {
        self.width
    }

    pub fn height(&self) -> Option<Dimension> {
        self.height
    }

    pub fn padding(&self) -> Edges<Option<Length>> {
        self.padding
    }

    pub fn margin(&self) -> Edges<Option<Length>> {
        self.margin
    }

    pub fn border_width(&self) -> Edges<Option<Length>> {
        self.border_width
    }

    pub fn gap(&self) -> Option<Length> {
        self.gap
    }

    pub fn overflow(&self) -> Option<Overflow> {
        self.overflow
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VisualStyleDeclaration {
    background: Option<Color>,
    border_color: Option<Color>,
    corner_radius: CornerRadii<Option<Length>>,
    opacity: Option<Opacity>,
}

impl VisualStyleDeclaration {
    pub fn background(&self) -> Option<Color> {
        self.background
    }

    pub fn border_color(&self) -> Option<Color> {
        self.border_color
    }

    pub fn corner_radius(&self) -> CornerRadii<Option<Length>> {
        self.corner_radius
    }

    pub fn opacity(&self) -> Option<Opacity> {
        self.opacity
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextStyleDeclaration {
    font_size: Option<Length>,
    text_color: Option<Color>,
    max_lines: Option<usize>,
    text_overflow: Option<TextOverflow>,
}

impl TextStyleDeclaration {
    pub fn font_size(&self) -> Option<Length> {
        self.font_size
    }

    pub fn text_color(&self) -> Option<Color> {
        self.text_color
    }

    pub fn max_lines(&self) -> Option<usize> {
        self.max_lines
    }

    pub fn text_overflow(&self) -> Option<TextOverflow> {
        self.text_overflow
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StyleDeclaration {
    layout: LayoutStyleDeclaration,
    visual: VisualStyleDeclaration,
    text: TextStyleDeclaration,
}

impl StyleDeclaration {
    pub fn layout(&self) -> &LayoutStyleDeclaration {
        &self.layout
    }

    pub fn visual(&self) -> &VisualStyleDeclaration {
        &self.visual
    }

    pub fn text(&self) -> &TextStyleDeclaration {
        &self.text
    }

    pub(crate) fn validate_inputs(&self) -> NekoResult<()> {
        if let Some(width) = self.layout.width {
            width.validate_for_layout("width")?;
        }
        if let Some(height) = self.layout.height {
            height.validate_for_layout("height")?;
        }
        validate_optional_length_edges(self.layout.padding, "padding")?;
        validate_optional_length_edges(self.layout.margin, "margin")?;
        validate_optional_length_edges(self.layout.border_width, "border width")?;
        if let Some(gap) = self.layout.gap {
            gap.validate_non_negative("gap")?;
        }
        if let Some(font_size) = self.text.font_size {
            font_size.validate_non_negative("font size")?;
        }
        if let Some(opacity) = self.visual.opacity {
            opacity.validate_input()?;
        }
        if let Some(background) = self.visual.background {
            background.validate_input()?;
        }
        if let Some(border_color) = self.visual.border_color {
            border_color.validate_input()?;
        }
        validate_optional_corner_radii(self.visual.corner_radius, "corner radius")?;
        if let Some(text_color) = self.text.text_color {
            text_color.validate_input()?;
        }
        Ok(())
    }

    pub fn display(mut self, value: Display) -> Self {
        self.set_display(value);
        self
    }

    pub fn padding(mut self, value: impl Into<Length>) -> Self {
        self.set_padding(value.into());
        self
    }

    pub fn padding_left(mut self, value: impl Into<Length>) -> Self {
        self.set_padding_left(value.into());
        self
    }

    pub fn margin(mut self, value: impl Into<Length>) -> Self {
        self.set_margin(value.into());
        self
    }

    pub fn margin_left(mut self, value: impl Into<Length>) -> Self {
        self.set_margin_left(value.into());
        self
    }

    pub fn border_width(mut self, value: impl Into<Length>) -> Self {
        self.set_border_width(value.into());
        self
    }

    pub fn gap(mut self, value: impl Into<Length>) -> Self {
        self.set_gap(value.into());
        self
    }

    pub fn overflow(mut self, value: Overflow) -> Self {
        self.set_overflow(value);
        self
    }

    pub fn width(mut self, value: impl Into<Dimension>) -> Self {
        self.set_width(value.into());
        self
    }

    pub fn height(mut self, value: impl Into<Dimension>) -> Self {
        self.set_height(value.into());
        self
    }

    pub fn background(mut self, value: Color) -> Self {
        self.set_background(value);
        self
    }

    pub fn border_color(mut self, value: Color) -> Self {
        self.set_border_color(value);
        self
    }

    pub fn border(mut self, width: impl Into<Length>, color: Color) -> Self {
        self.set_border(width.into(), color);
        self
    }

    pub fn corner_radius(mut self, value: impl Into<Length>) -> Self {
        self.set_corner_radius(value.into());
        self
    }

    pub fn radius(self, value: impl Into<Length>) -> Self {
        self.corner_radius(value)
    }

    pub fn rounded(self, value: impl Into<Length>) -> Self {
        self.corner_radius(value)
    }

    pub fn opacity(mut self, value: impl Into<Opacity>) -> Self {
        self.set_opacity(value.into());
        self
    }

    pub fn font_size(mut self, value: impl Into<Length>) -> Self {
        self.set_font_size(value.into());
        self
    }

    pub fn text_color(mut self, value: Color) -> Self {
        self.set_text_color(value);
        self
    }

    pub fn line_clamp(mut self, lines: usize) -> Self {
        self.set_line_clamp(lines);
        self
    }

    pub(crate) fn set_display(&mut self, value: Display) {
        self.layout.display = Some(value);
    }

    pub(crate) fn set_padding(&mut self, value: Length) {
        self.layout.padding = Edges::all(Some(value));
    }

    pub(crate) fn set_padding_x(&mut self, value: Length) {
        self.layout.padding.left = Some(value);
        self.layout.padding.right = Some(value);
    }

    pub(crate) fn set_padding_y(&mut self, value: Length) {
        self.layout.padding.top = Some(value);
        self.layout.padding.bottom = Some(value);
    }

    pub(crate) fn set_padding_top(&mut self, value: Length) {
        self.layout.padding.top = Some(value);
    }

    pub(crate) fn set_padding_right(&mut self, value: Length) {
        self.layout.padding.right = Some(value);
    }

    pub(crate) fn set_padding_bottom(&mut self, value: Length) {
        self.layout.padding.bottom = Some(value);
    }

    pub(crate) fn set_padding_left(&mut self, value: Length) {
        self.layout.padding.left = Some(value);
    }

    pub(crate) fn set_margin(&mut self, value: Length) {
        self.layout.margin = Edges::all(Some(value));
    }

    pub(crate) fn set_margin_x(&mut self, value: Length) {
        self.layout.margin.left = Some(value);
        self.layout.margin.right = Some(value);
    }

    pub(crate) fn set_margin_y(&mut self, value: Length) {
        self.layout.margin.top = Some(value);
        self.layout.margin.bottom = Some(value);
    }

    pub(crate) fn set_margin_top(&mut self, value: Length) {
        self.layout.margin.top = Some(value);
    }

    pub(crate) fn set_margin_right(&mut self, value: Length) {
        self.layout.margin.right = Some(value);
    }

    pub(crate) fn set_margin_bottom(&mut self, value: Length) {
        self.layout.margin.bottom = Some(value);
    }

    pub(crate) fn set_margin_left(&mut self, value: Length) {
        self.layout.margin.left = Some(value);
    }

    pub(crate) fn set_border_width(&mut self, value: Length) {
        self.layout.border_width = Edges::all(Some(value));
    }

    pub(crate) fn set_gap(&mut self, value: Length) {
        self.layout.gap = Some(value);
    }

    pub(crate) fn set_overflow(&mut self, value: Overflow) {
        self.layout.overflow = Some(value);
    }

    pub(crate) fn set_width(&mut self, value: Dimension) {
        self.layout.width = Some(value);
    }

    pub(crate) fn set_height(&mut self, value: Dimension) {
        self.layout.height = Some(value);
    }

    pub(crate) fn set_background(&mut self, value: Color) {
        self.visual.background = Some(value);
    }

    pub(crate) fn set_border_color(&mut self, value: Color) {
        self.visual.border_color = Some(value);
    }

    pub(crate) fn set_border(&mut self, width: Length, color: Color) {
        self.set_border_width(width);
        self.set_border_color(color);
    }

    pub(crate) fn set_corner_radius(&mut self, value: Length) {
        self.visual.corner_radius = CornerRadii::all(Some(value));
    }

    pub(crate) fn set_opacity(&mut self, value: Opacity) {
        self.visual.opacity = Some(value);
    }

    pub(crate) fn set_font_size(&mut self, value: Length) {
        self.text.font_size = Some(value);
    }

    pub(crate) fn set_text_color(&mut self, value: Color) {
        self.text.text_color = Some(value);
    }

    pub(crate) fn set_line_clamp(&mut self, lines: usize) {
        self.text.max_lines = Some(lines);
        self.text.text_overflow = Some(TextOverflow::Ellipsis);
    }

    pub fn dirty_lanes_since(&self, old: &Self) -> DirtyLanes {
        let mut lanes = DirtyLanes::empty();

        if self.layout != old.layout {
            lanes.insert(DirtyLane::Style.flag());
            lanes.insert(DirtyLane::Layout.flag());
            lanes.insert(DirtyLane::Semantics.flag());
            lanes.insert(DirtyLane::Paint.flag());
        }

        if self.visual != old.visual {
            lanes.insert(DirtyLane::Style.flag());
            lanes.insert(DirtyLane::Paint.flag());
        }

        if self.text.text_color != old.text.text_color {
            lanes.insert(DirtyLane::Style.flag());
            lanes.insert(DirtyLane::Paint.flag());
        }

        if self.text.font_size != old.text.font_size
            || self.text.max_lines != old.text.max_lines
            || self.text.text_overflow != old.text.text_overflow
        {
            lanes.insert(DirtyLane::Style.flag());
            lanes.insert(DirtyLane::Text.flag());
            lanes.insert(DirtyLane::Layout.flag());
            lanes.insert(DirtyLane::Semantics.flag());
            lanes.insert(DirtyLane::Paint.flag());
        }

        lanes
    }
}

fn validate_optional_length_edges(
    edges: Edges<Option<Length>>,
    label: &'static str,
) -> NekoResult<()> {
    if let Some(top) = edges.top {
        top.validate_non_negative(label)?;
    }
    if let Some(right) = edges.right {
        right.validate_non_negative(label)?;
    }
    if let Some(bottom) = edges.bottom {
        bottom.validate_non_negative(label)?;
    }
    if let Some(left) = edges.left {
        left.validate_non_negative(label)?;
    }
    Ok(())
}

fn validate_optional_corner_radii(
    radii: CornerRadii<Option<Length>>,
    label: &'static str,
) -> NekoResult<()> {
    if let Some(top_left) = radii.top_left {
        top_left.validate_non_negative(label)?;
    }
    if let Some(top_right) = radii.top_right {
        top_right.validate_non_negative(label)?;
    }
    if let Some(bottom_right) = radii.bottom_right {
        bottom_right.validate_non_negative(label)?;
    }
    if let Some(bottom_left) = radii.bottom_left {
        bottom_left.validate_non_negative(label)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::DirtyLane;
    use crate::style::{
        Color, CornerRadii, Display, Edges, Overflow, StyleDeclaration, TextOverflow, fill, px,
    };

    #[test]
    fn canonical_and_alias_expansion_is_longhand_ordered() {
        let style = StyleDeclaration::default()
            .padding(px(8.0))
            .padding_left(px(4.0))
            .margin(px(10.0))
            .margin_left(px(6.0))
            .border_width(px(2.0))
            .border(px(3.0), Color::rgb(8, 9, 10))
            .corner_radius(px(5.0))
            .radius(px(6.0))
            .rounded(px(7.0))
            .gap(px(3.0))
            .overflow(Overflow::Scroll);
        let padding = style.layout().padding();
        let margin = style.layout().margin();

        assert_eq!(padding.top, Some(px(8.0)));
        assert_eq!(padding.right, Some(px(8.0)));
        assert_eq!(padding.bottom, Some(px(8.0)));
        assert_eq!(padding.left, Some(px(4.0)));
        assert_eq!(margin.top, Some(px(10.0)));
        assert_eq!(margin.left, Some(px(6.0)));
        assert_eq!(style.layout().border_width(), Edges::all(Some(px(3.0))));
        assert_eq!(style.visual().border_color(), Some(Color::rgb(8, 9, 10)));
        assert_eq!(
            style.visual().corner_radius(),
            CornerRadii::all(Some(px(7.0)))
        );
        assert_eq!(style.layout().gap(), Some(px(3.0)));
        assert_eq!(style.layout().overflow(), Some(Overflow::Scroll));
    }

    #[test]
    fn line_clamp_is_canonical_text_policy() {
        let style = StyleDeclaration::default().line_clamp(3);

        assert_eq!(style.text().max_lines(), Some(3));
        assert_eq!(style.text().text_overflow(), Some(TextOverflow::Ellipsis));
    }

    #[test]
    fn dirty_classification_is_facet_specific() {
        let base = StyleDeclaration::default();
        let visual = base.clone().background(Color::rgb(1, 2, 3)).opacity(0.5);
        let layout = base.clone().width(fill()).gap(px(8.0));
        let border_layout = base.clone().border_width(px(2.0));
        let border_visual = base
            .clone()
            .border_color(Color::rgb(4, 5, 6))
            .rounded(px(6.0));
        let text = base.clone().font_size(px(18.0));

        let visual_lanes = visual.dirty_lanes_since(&base);
        assert!(visual_lanes.contains(DirtyLane::Style.flag()));
        assert!(visual_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!visual_lanes.contains(DirtyLane::Layout.flag()));
        assert!(!visual_lanes.contains(DirtyLane::Text.flag()));

        let layout_lanes = layout.dirty_lanes_since(&base);
        assert!(layout_lanes.contains(DirtyLane::Layout.flag()));
        assert!(layout_lanes.contains(DirtyLane::Semantics.flag()));
        assert!(!layout_lanes.contains(DirtyLane::Text.flag()));

        let border_layout_lanes = border_layout.dirty_lanes_since(&base);
        assert!(border_layout_lanes.contains(DirtyLane::Layout.flag()));
        assert!(border_layout_lanes.contains(DirtyLane::Semantics.flag()));
        assert!(border_layout_lanes.contains(DirtyLane::Paint.flag()));

        let border_visual_lanes = border_visual.dirty_lanes_since(&base);
        assert!(border_visual_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!border_visual_lanes.contains(DirtyLane::Layout.flag()));

        let text_lanes = text.dirty_lanes_since(&base);
        assert!(text_lanes.contains(DirtyLane::Text.flag()));
        assert!(text_lanes.contains(DirtyLane::Layout.flag()));
        assert!(text_lanes.contains(DirtyLane::Semantics.flag()));

        let text_color = base.clone().text_color(Color::rgb(10, 20, 30));
        let text_color_lanes = text_color.dirty_lanes_since(&base);
        assert!(text_color_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!text_color_lanes.contains(DirtyLane::Text.flag()));
        assert!(!text_color_lanes.contains(DirtyLane::Semantics.flag()));

        let display_none = base.clone().display(Display::None);
        let display_lanes = display_none.dirty_lanes_since(&base);
        assert!(display_lanes.contains(DirtyLane::Layout.flag()));
        assert!(display_lanes.contains(DirtyLane::Semantics.flag()));
    }
}
