use crate::element::{Div, Element, Input, Text};
use crate::style::{Color, Dimension, Display, Length, Opacity, Overflow, StyleDeclaration};

pub trait StyleExt: Sized {
    fn update_style<F>(self, update: F) -> Self
    where
        F: FnOnce(&mut StyleDeclaration);

    fn display(self, value: Display) -> Self {
        self.update_style(|style| style.set_display(value))
    }

    fn padding(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding(value))
    }

    fn p(self, value: impl Into<Length>) -> Self {
        self.padding(value)
    }

    fn padding_x(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_x(value))
    }

    fn px(self, value: impl Into<Length>) -> Self {
        self.padding_x(value)
    }

    fn padding_y(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_y(value))
    }

    fn py(self, value: impl Into<Length>) -> Self {
        self.padding_y(value)
    }

    fn padding_top(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_top(value))
    }

    fn pt(self, value: impl Into<Length>) -> Self {
        self.padding_top(value)
    }

    fn padding_right(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_right(value))
    }

    fn pr(self, value: impl Into<Length>) -> Self {
        self.padding_right(value)
    }

    fn padding_bottom(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_bottom(value))
    }

    fn pb(self, value: impl Into<Length>) -> Self {
        self.padding_bottom(value)
    }

    fn padding_left(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_padding_left(value))
    }

    fn pl(self, value: impl Into<Length>) -> Self {
        self.padding_left(value)
    }

    fn margin(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin(value))
    }

    fn m(self, value: impl Into<Length>) -> Self {
        self.margin(value)
    }

    fn margin_x(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_x(value))
    }

    fn mx(self, value: impl Into<Length>) -> Self {
        self.margin_x(value)
    }

    fn margin_y(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_y(value))
    }

    fn my(self, value: impl Into<Length>) -> Self {
        self.margin_y(value)
    }

    fn margin_top(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_top(value))
    }

    fn mt(self, value: impl Into<Length>) -> Self {
        self.margin_top(value)
    }

    fn margin_right(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_right(value))
    }

    fn mr(self, value: impl Into<Length>) -> Self {
        self.margin_right(value)
    }

    fn margin_bottom(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_bottom(value))
    }

    fn mb(self, value: impl Into<Length>) -> Self {
        self.margin_bottom(value)
    }

    fn margin_left(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_margin_left(value))
    }

    fn ml(self, value: impl Into<Length>) -> Self {
        self.margin_left(value)
    }

    fn border_width(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_border_width(value))
    }

    fn gap(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_gap(value))
    }

    fn overflow(self, value: Overflow) -> Self {
        self.update_style(|style| style.set_overflow(value))
    }

    fn width(self, value: impl Into<Dimension>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_width(value))
    }

    fn w(self, value: impl Into<Dimension>) -> Self {
        self.width(value)
    }

    fn height(self, value: impl Into<Dimension>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_height(value))
    }

    fn h(self, value: impl Into<Dimension>) -> Self {
        self.height(value)
    }

    fn background(self, value: Color) -> Self {
        self.update_style(|style| style.set_background(value))
    }

    fn bg(self, value: Color) -> Self {
        self.background(value)
    }

    fn border_color(self, value: Color) -> Self {
        self.update_style(|style| style.set_border_color(value))
    }

    fn border(self, width: impl Into<Length>, color: Color) -> Self {
        let width = width.into();
        self.update_style(|style| style.set_border(width, color))
    }

    fn corner_radius(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_corner_radius(value))
    }

    fn radius(self, value: impl Into<Length>) -> Self {
        self.corner_radius(value)
    }

    fn rounded(self, value: impl Into<Length>) -> Self {
        self.corner_radius(value)
    }

    fn opacity(self, value: impl Into<Opacity>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_opacity(value))
    }

    fn font_size(self, value: impl Into<Length>) -> Self {
        let value = value.into();
        self.update_style(|style| style.set_font_size(value))
    }

    fn text_color(self, value: Color) -> Self {
        self.update_style(|style| style.set_text_color(value))
    }

    fn line_clamp(self, lines: usize) -> Self {
        self.update_style(|style| style.set_line_clamp(lines))
    }
}

impl StyleExt for Div {
    fn update_style<F>(mut self, update: F) -> Self
    where
        F: FnOnce(&mut StyleDeclaration),
    {
        update(self.style_mut());
        self
    }
}

impl StyleExt for Text {
    fn update_style<F>(mut self, update: F) -> Self
    where
        F: FnOnce(&mut StyleDeclaration),
    {
        update(self.style_mut());
        self
    }
}

impl StyleExt for Input {
    fn update_style<F>(mut self, update: F) -> Self
    where
        F: FnOnce(&mut StyleDeclaration),
    {
        update(self.style_mut());
        self
    }
}
impl StyleExt for Element {
    fn update_style<F>(mut self, update: F) -> Self
    where
        F: FnOnce(&mut StyleDeclaration),
    {
        update(self.style_mut());
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::element::{IntoElement, div};
    use crate::style::{Color, CornerRadii, Dimension, Edges, Overflow, StyleExt, fill, px};

    #[test]
    fn style_extension_methods_write_canonical_declaration() {
        let element = div()
            .padding(px(12.0))
            .p(px(8.0))
            .pl(px(4.0))
            .margin(px(16.0))
            .m(px(10.0))
            .ml(px(6.0))
            .border_width(px(2.0))
            .border_color(Color::rgb(1, 1, 1))
            .border(px(3.0), Color::rgb(7, 8, 9))
            .corner_radius(px(4.0))
            .radius(px(5.0))
            .rounded(px(6.0))
            .gap(px(2.0))
            .overflow(Overflow::Scroll)
            .width(fill())
            .w(fill())
            .background(Color::rgb(1, 2, 3))
            .bg(Color::rgb(4, 5, 6))
            .opacity(0.75)
            .into_element();

        let padding = element.style().layout().padding();
        let margin = element.style().layout().margin();
        assert_eq!(padding.top, Some(px(8.0)));
        assert_eq!(padding.left, Some(px(4.0)));
        assert_eq!(margin.top, Some(px(10.0)));
        assert_eq!(margin.left, Some(px(6.0)));
        assert_eq!(
            element.style().layout().border_width(),
            Edges::all(Some(px(3.0)))
        );
        assert_eq!(
            element.style().visual().border_color(),
            Some(Color::rgb(7, 8, 9))
        );
        assert_eq!(
            element.style().visual().corner_radius(),
            CornerRadii::all(Some(px(6.0)))
        );
        assert_eq!(element.style().layout().gap(), Some(px(2.0)));
        assert_eq!(element.style().layout().overflow(), Some(Overflow::Scroll));
        assert_eq!(element.style().layout().width(), Some(Dimension::Fill));
        assert_eq!(
            element.style().visual().background(),
            Some(Color::rgb(4, 5, 6))
        );
        assert_eq!(
            element
                .style()
                .visual()
                .opacity()
                .map(|value| value.as_f32()),
            Some(0.75)
        );
    }
}
