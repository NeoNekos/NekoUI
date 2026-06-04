use crate::diagnostic::{DirtyLane, DirtyLanes};
use crate::style::{
    Color, CornerRadii, Dimension, Display, Edges, Length, Opacity, Overflow, StyleDeclaration,
    TextOverflow,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLayoutStyle {
    display: Display,
    width: Dimension,
    height: Dimension,
    padding: Edges<Length>,
    margin: Edges<Length>,
    border_width: Edges<Length>,
    gap: Length,
    overflow: Overflow,
}

impl Default for ResolvedLayoutStyle {
    fn default() -> Self {
        Self {
            display: Display::Block,
            width: Dimension::Auto,
            height: Dimension::Auto,
            padding: Edges::all(Length::ZERO),
            margin: Edges::all(Length::ZERO),
            border_width: Edges::all(Length::ZERO),
            gap: Length::ZERO,
            overflow: Overflow::Visible,
        }
    }
}

impl ResolvedLayoutStyle {
    pub fn display(&self) -> Display {
        self.display
    }

    pub fn width(&self) -> Dimension {
        self.width
    }

    pub fn height(&self) -> Dimension {
        self.height
    }

    pub fn padding(&self) -> Edges<Length> {
        self.padding
    }

    pub fn margin(&self) -> Edges<Length> {
        self.margin
    }

    pub fn border_width(&self) -> Edges<Length> {
        self.border_width
    }

    pub fn gap(&self) -> Length {
        self.gap
    }

    pub fn overflow(&self) -> Overflow {
        self.overflow
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedVisualStyle {
    background: Option<Color>,
    border_color: Option<Color>,
    corner_radius: CornerRadii<Length>,
    opacity: Opacity,
}

impl Default for ResolvedVisualStyle {
    fn default() -> Self {
        Self {
            background: None,
            border_color: None,
            corner_radius: CornerRadii::all(Length::ZERO),
            opacity: Opacity::OPAQUE,
        }
    }
}

impl ResolvedVisualStyle {
    pub fn background(&self) -> Option<Color> {
        self.background
    }

    pub fn border_color(&self) -> Option<Color> {
        self.border_color
    }

    pub fn corner_radius(&self) -> CornerRadii<Length> {
        self.corner_radius
    }

    pub fn opacity(&self) -> Opacity {
        self.opacity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedTextStyle {
    font_size: Length,
    text_color: Color,
    max_lines: Option<usize>,
    text_overflow: TextOverflow,
}

impl Default for ResolvedTextStyle {
    fn default() -> Self {
        Self {
            font_size: Length::px(14.0),
            text_color: Color::BLACK,
            max_lines: None,
            text_overflow: TextOverflow::Clip,
        }
    }
}

impl ResolvedTextStyle {
    pub fn font_size(&self) -> Length {
        self.font_size
    }

    pub fn text_color(&self) -> Color {
        self.text_color
    }

    pub fn max_lines(&self) -> Option<usize> {
        self.max_lines
    }

    pub fn text_overflow(&self) -> TextOverflow {
        self.text_overflow
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedStyle {
    layout: ResolvedLayoutStyle,
    visual: ResolvedVisualStyle,
    text: ResolvedTextStyle,
}

impl ResolvedStyle {
    pub fn resolve(declaration: &StyleDeclaration, parent: Option<&ResolvedStyle>) -> Self {
        let default_text = ResolvedTextStyle::default();
        let parent_text = parent.map(|style| &style.text);
        let layout_declaration = declaration.layout();
        let visual_declaration = declaration.visual();
        let text_declaration = declaration.text();
        let padding = layout_declaration.padding();
        let margin = layout_declaration.margin();
        let border_width = layout_declaration.border_width();
        let corner_radius = visual_declaration.corner_radius();

        Self {
            layout: ResolvedLayoutStyle {
                display: layout_declaration.display().unwrap_or(Display::Block),
                width: layout_declaration.width().unwrap_or(Dimension::Auto),
                height: layout_declaration.height().unwrap_or(Dimension::Auto),
                padding: Edges {
                    top: padding.top.unwrap_or(Length::ZERO),
                    right: padding.right.unwrap_or(Length::ZERO),
                    bottom: padding.bottom.unwrap_or(Length::ZERO),
                    left: padding.left.unwrap_or(Length::ZERO),
                },
                margin: Edges {
                    top: margin.top.unwrap_or(Length::ZERO),
                    right: margin.right.unwrap_or(Length::ZERO),
                    bottom: margin.bottom.unwrap_or(Length::ZERO),
                    left: margin.left.unwrap_or(Length::ZERO),
                },
                border_width: Edges {
                    top: border_width.top.unwrap_or(Length::ZERO),
                    right: border_width.right.unwrap_or(Length::ZERO),
                    bottom: border_width.bottom.unwrap_or(Length::ZERO),
                    left: border_width.left.unwrap_or(Length::ZERO),
                },
                gap: layout_declaration.gap().unwrap_or(Length::ZERO),
                overflow: layout_declaration.overflow().unwrap_or(Overflow::Visible),
            },
            visual: ResolvedVisualStyle {
                background: visual_declaration.background(),
                border_color: visual_declaration.border_color(),
                corner_radius: CornerRadii {
                    top_left: corner_radius.top_left.unwrap_or(Length::ZERO),
                    top_right: corner_radius.top_right.unwrap_or(Length::ZERO),
                    bottom_right: corner_radius.bottom_right.unwrap_or(Length::ZERO),
                    bottom_left: corner_radius.bottom_left.unwrap_or(Length::ZERO),
                },
                opacity: visual_declaration.opacity().unwrap_or_default(),
            },
            text: ResolvedTextStyle {
                font_size: text_declaration
                    .font_size()
                    .or_else(|| parent_text.map(|text| text.font_size))
                    .unwrap_or(default_text.font_size),
                text_color: text_declaration
                    .text_color()
                    .or_else(|| parent_text.map(|text| text.text_color))
                    .unwrap_or(default_text.text_color),
                max_lines: text_declaration.max_lines(),
                text_overflow: text_declaration
                    .text_overflow()
                    .unwrap_or(TextOverflow::Clip),
            },
        }
    }

    pub fn layout(&self) -> &ResolvedLayoutStyle {
        &self.layout
    }

    pub fn visual(&self) -> &ResolvedVisualStyle {
        &self.visual
    }

    pub fn text(&self) -> &ResolvedTextStyle {
        &self.text
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

#[cfg(test)]
mod tests {
    use crate::diagnostic::DirtyLane;
    use crate::style::{Color, CornerRadii, Edges, Overflow, ResolvedStyle, StyleDeclaration, px};

    #[test]
    fn text_style_inherits_but_box_geometry_defaults() {
        let parent = ResolvedStyle::resolve(
            &StyleDeclaration::default()
                .font_size(px(18.0))
                .text_color(Color::rgb(20, 30, 40))
                .gap(px(6.0))
                .padding(px(12.0))
                .border_width(px(2.0))
                .border_color(Color::rgb(1, 2, 3))
                .rounded(px(4.0)),
            None,
        );
        let child = ResolvedStyle::resolve(&StyleDeclaration::default(), Some(&parent));

        assert_eq!(child.text().font_size(), px(18.0));
        assert_eq!(child.text().text_color(), Color::rgb(20, 30, 40));
        assert_eq!(child.layout().padding().left, px(0.0));
        assert_eq!(child.layout().border_width(), Edges::all(px(0.0)));
        assert_eq!(child.layout().gap(), px(0.0));
        assert_eq!(child.visual().border_color(), None);
        assert_eq!(child.visual().corner_radius(), CornerRadii::all(px(0.0)));
        assert_eq!(child.layout().overflow(), Overflow::Visible);
    }

    #[test]
    fn dirty_classification_tracks_semantic_output_changes() {
        let base = ResolvedStyle::resolve(&StyleDeclaration::default(), None);
        let text_color = ResolvedStyle::resolve(
            &StyleDeclaration::default().text_color(Color::rgb(1, 2, 3)),
            None,
        );
        let text_geometry =
            ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(20.0)), None);
        let layout = ResolvedStyle::resolve(&StyleDeclaration::default().padding(px(4.0)), None);
        let border_layout =
            ResolvedStyle::resolve(&StyleDeclaration::default().border_width(px(2.0)), None);
        let border_visual = ResolvedStyle::resolve(
            &StyleDeclaration::default()
                .border_color(Color::rgb(1, 2, 3))
                .rounded(px(4.0)),
            None,
        );
        let overflow = ResolvedStyle::resolve(
            &StyleDeclaration::default().overflow(Overflow::Scroll),
            None,
        );

        let text_color_lanes = text_color.dirty_lanes_since(&base);
        assert!(text_color_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!text_color_lanes.contains(DirtyLane::Semantics.flag()));

        let text_geometry_lanes = text_geometry.dirty_lanes_since(&base);
        assert!(text_geometry_lanes.contains(DirtyLane::Text.flag()));
        assert!(text_geometry_lanes.contains(DirtyLane::Semantics.flag()));

        let layout_lanes = layout.dirty_lanes_since(&base);
        assert!(layout_lanes.contains(DirtyLane::Layout.flag()));
        assert!(layout_lanes.contains(DirtyLane::Semantics.flag()));

        let border_layout_lanes = border_layout.dirty_lanes_since(&base);
        assert!(border_layout_lanes.contains(DirtyLane::Layout.flag()));
        assert!(border_layout_lanes.contains(DirtyLane::Semantics.flag()));
        assert!(border_layout_lanes.contains(DirtyLane::Paint.flag()));

        let border_visual_lanes = border_visual.dirty_lanes_since(&base);
        assert!(border_visual_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!border_visual_lanes.contains(DirtyLane::Layout.flag()));

        let overflow_lanes = overflow.dirty_lanes_since(&base);
        assert!(overflow_lanes.contains(DirtyLane::Layout.flag()));
        assert!(overflow_lanes.contains(DirtyLane::Semantics.flag()));
        assert!(overflow_lanes.contains(DirtyLane::Paint.flag()));
    }
}
