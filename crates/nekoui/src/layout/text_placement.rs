use crate::element::ElementKind;
use crate::layout::LayoutRect;
use crate::text::TextLayoutRef;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextVisualPlacement {
    rect: LayoutRect,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextViewportPlacement {
    text_draw_rect: LayoutRect,
    visible_caret_rect: LayoutRect,
    viewport_rect: LayoutRect,
    input_inline_scroll: f32,
}

impl TextVisualPlacement {
    pub(crate) fn rect(self) -> LayoutRect {
        self.rect
    }

    pub(crate) fn place_local_rect(self, rect: LayoutRect) -> LayoutRect {
        rect.translate(self.rect.x(), self.rect.y())
    }
}

impl TextViewportPlacement {
    pub(crate) fn text_draw_rect(self) -> LayoutRect {
        self.text_draw_rect
    }

    pub(crate) fn visible_caret_rect(self) -> LayoutRect {
        self.visible_caret_rect
    }

    pub(crate) fn viewport_rect(self) -> LayoutRect {
        self.viewport_rect
    }

    #[cfg(test)]
    pub(crate) fn input_inline_scroll(self) -> f32 {
        self.input_inline_scroll
    }
}

pub(crate) fn text_visual_placement(
    kind: ElementKind,
    content_rect: LayoutRect,
    text_layout: &TextLayoutRef,
) -> TextVisualPlacement {
    let dy = if kind == ElementKind::Input {
        ((content_rect.height() - text_layout.metrics().height) / 2.0).max(0.0)
    } else {
        0.0
    };
    TextVisualPlacement {
        rect: content_rect.translate(0.0, dy),
    }
}

pub(crate) fn text_viewport_placement(
    kind: ElementKind,
    content_rect: LayoutRect,
    text_layout: &TextLayoutRef,
) -> TextViewportPlacement {
    let visual = text_visual_placement(kind, content_rect, text_layout);
    let trailing_caret = text_layout.trailing_caret_rect();
    let input_inline_scroll = if kind == ElementKind::Input {
        derived_input_inline_scroll(content_rect, trailing_caret)
    } else {
        0.0
    };
    let text_draw_rect = visual.rect().translate(-input_inline_scroll, 0.0);
    let unclamped_caret = visual
        .place_local_rect(trailing_caret)
        .translate(-input_inline_scroll, 0.0);
    let visible_caret_rect = clamp_input_caret_to_viewport(kind, unclamped_caret, content_rect);

    TextViewportPlacement {
        text_draw_rect,
        visible_caret_rect,
        viewport_rect: content_rect,
        input_inline_scroll,
    }
}

fn clamp_input_caret_to_viewport(
    kind: ElementKind,
    caret: LayoutRect,
    content_rect: LayoutRect,
) -> LayoutRect {
    if kind != ElementKind::Input {
        return caret;
    }
    let caret_width = caret.width().min(content_rect.width()).max(0.0);
    let min_x = content_rect.x();
    let max_x = content_rect.x() + (content_rect.width() - caret_width).max(0.0);
    let x = caret.x().clamp(min_x, max_x);
    LayoutRect::new(x, caret.y(), caret_width, caret.height())
}

fn derived_input_inline_scroll(content_rect: LayoutRect, trailing_caret: LayoutRect) -> f32 {
    let trailing_caret_right = trailing_caret.x() + trailing_caret.width();
    let content_width = content_rect.width();
    if !trailing_caret_right.is_finite() || !content_width.is_finite() {
        return 0.0;
    }
    (trailing_caret_right - content_width).max(0.0)
}

#[cfg(test)]
mod tests {
    use crate::element::ElementKind;
    use crate::layout::LayoutRect;

    use super::clamp_input_caret_to_viewport;

    #[test]
    fn input_caret_clamps_to_zero_width_viewport() {
        let viewport = LayoutRect::new(10.0, 20.0, 0.0, 16.0);
        let caret = LayoutRect::new(50.0, 20.0, 1.0, 16.0);

        let clamped = clamp_input_caret_to_viewport(ElementKind::Input, caret, viewport);

        assert_eq!(clamped, LayoutRect::new(10.0, 20.0, 0.0, 16.0));
    }

    #[test]
    fn input_caret_clamps_to_tiny_viewport() {
        let viewport = LayoutRect::new(10.0, 20.0, 0.5, 16.0);
        let caret = LayoutRect::new(50.0, 20.0, 1.0, 16.0);

        let clamped = clamp_input_caret_to_viewport(ElementKind::Input, caret, viewport);

        assert_eq!(clamped, LayoutRect::new(10.0, 20.0, 0.5, 16.0));
    }

    #[test]
    fn ordinary_text_caret_is_not_clamped_to_viewport() {
        let viewport = LayoutRect::new(10.0, 20.0, 0.5, 16.0);
        let caret = LayoutRect::new(50.0, 20.0, 1.0, 16.0);

        let clamped = clamp_input_caret_to_viewport(ElementKind::Text, caret, viewport);

        assert_eq!(clamped, caret);
    }
}
