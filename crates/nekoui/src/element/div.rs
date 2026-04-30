use crate::style::{
    Absolute, AlignItems, AlignSelf, Background, Border, BoxSizing, Color, CornerRadii, Definite,
    Direction, Display, EdgeWidths, Edges, FlexDirection, FlexWrap, FontStyle, FontWeight, Gap,
    IntoFontFamilies, JustifyContent, LayoutSize, Length, Overflow, Size, Style, TextAlign,
    WhiteSpace,
};

use super::builder_macros::{
    impl_shared_flex_item_builders, impl_shared_key_size_margin_builders,
    impl_shared_text_style_builders, impl_shared_window_chrome_builders,
};
use super::{
    AnyElement, Fragment, InteractionState, IntoElement, IntoElements, ParentElement,
    WindowFrameArea,
};
use crate::input::{FocusPolicy, TextInputPurpose, TextInputState};
use crate::semantics::{SemanticsRole, SemanticsState};

#[derive(Debug, Clone, PartialEq)]
pub struct Div {
    pub(crate) key: Option<u64>,
    pub(crate) style: Style,
    pub(crate) window_frame_area: Option<WindowFrameArea>,
    pub(crate) interaction: InteractionState,
    pub(crate) semantics: SemanticsState,
    pub(crate) children: Fragment,
}

pub fn div() -> Div {
    Div {
        key: None,
        style: Style::default(),
        window_frame_area: None,
        interaction: InteractionState::default(),
        semantics: SemanticsState::default(),
        children: Fragment::new(),
    }
}

impl Default for Div {
    fn default() -> Self {
        div()
    }
}

impl Div {
    impl_shared_key_size_margin_builders!();

    pub fn padding(mut self, padding: impl Into<Edges<Definite>>) -> Self {
        self.style.layout.padding = padding.into();
        self
    }

    pub fn p(mut self, value: impl Into<Definite>) -> Self {
        self.style.layout.padding = Edges::all(value.into());
        self
    }

    pub fn px(mut self, value: impl Into<Definite>) -> Self {
        let value = value.into();
        self.style.layout.padding.left = value;
        self.style.layout.padding.right = value;
        self
    }

    pub fn py(mut self, value: impl Into<Definite>) -> Self {
        let value = value.into();
        self.style.layout.padding.top = value;
        self.style.layout.padding.bottom = value;
        self
    }

    pub fn pt(mut self, value: impl Into<Definite>) -> Self {
        self.style.layout.padding.top = value.into();
        self
    }

    pub fn pr(mut self, value: impl Into<Definite>) -> Self {
        self.style.layout.padding.right = value.into();
        self
    }

    pub fn pb(mut self, value: impl Into<Definite>) -> Self {
        self.style.layout.padding.bottom = value.into();
        self
    }

    pub fn pl(mut self, value: impl Into<Definite>) -> Self {
        self.style.layout.padding.left = value.into();
        self
    }

    pub fn mt(mut self, value: impl Into<Length>) -> Self {
        self.style.layout.margin.top = value.into();
        self
    }

    pub fn mr(mut self, value: impl Into<Length>) -> Self {
        self.style.layout.margin.right = value.into();
        self
    }

    pub fn mb(mut self, value: impl Into<Length>) -> Self {
        self.style.layout.margin.bottom = value.into();
        self
    }

    pub fn ml(mut self, value: impl Into<Length>) -> Self {
        self.style.layout.margin.left = value.into();
        self
    }

    pub fn flex_direction(mut self, direction: FlexDirection) -> Self {
        self.style.layout.flex_direction = direction;
        self
    }

    pub fn direction(self, direction: Direction) -> Self {
        self.flex_direction(direction)
    }

    pub fn flex_row(self) -> Self {
        self.flex_direction(FlexDirection::Row)
    }

    pub fn flex(self) -> Self {
        self.display(Display::Flex)
    }

    pub fn block(self) -> Self {
        self.display(Display::Block)
    }

    pub fn flex_col(self) -> Self {
        self.flex_direction(FlexDirection::Column)
    }

    pub fn flex_wrap(mut self, wrap: FlexWrap) -> Self {
        self.style.layout.flex_wrap = wrap;
        self
    }

    pub fn flex_nowrap(self) -> Self {
        self.flex_wrap(FlexWrap::NoWrap)
    }

    impl_shared_flex_item_builders!();

    pub fn flex_1(self) -> Self {
        self.flex_grow(1.0)
            .flex_shrink(1.0)
            .flex_basis(crate::style::Percent(0.0))
    }

    pub fn gap(mut self, gap: impl Into<Gap<Definite>>) -> Self {
        self.style.layout.gap = gap.into();
        self
    }

    pub fn gap_x(mut self, gap: impl Into<Definite>) -> Self {
        self.style.layout.gap.column = gap.into();
        self
    }

    pub fn gap_y(mut self, gap: impl Into<Definite>) -> Self {
        self.style.layout.gap.row = gap.into();
        self
    }

    pub fn justify_content(mut self, justify_content: JustifyContent) -> Self {
        self.style.layout.justify_content = justify_content;
        self
    }

    pub fn justify(self, justify_content: JustifyContent) -> Self {
        self.justify_content(justify_content)
    }

    pub fn justify_center(self) -> Self {
        self.justify_content(JustifyContent::Center)
    }

    pub fn justify_start(self) -> Self {
        self.justify_content(JustifyContent::Start)
    }

    pub fn justify_end(self) -> Self {
        self.justify_content(JustifyContent::End)
    }

    pub fn justify_between(self) -> Self {
        self.justify_content(JustifyContent::SpaceBetween)
    }

    pub fn align_items(mut self, align_items: AlignItems) -> Self {
        self.style.layout.align_items = align_items;
        self
    }

    pub fn items_center(self) -> Self {
        self.align_items(AlignItems::Center)
    }

    pub fn items_start(self) -> Self {
        self.align_items(AlignItems::Start)
    }

    pub fn items_end(self) -> Self {
        self.align_items(AlignItems::End)
    }

    pub fn display(mut self, display: Display) -> Self {
        self.style.layout.display = display;
        self
    }

    pub fn display_none(self) -> Self {
        self.display(Display::None)
    }

    pub fn hidden(self) -> Self {
        self.display_none()
    }

    pub fn background(mut self, background: impl Into<Background>) -> Self {
        self.style.paint.background = Some(background.into());
        self
    }

    pub fn bg(self, background: impl Into<Background>) -> Self {
        self.background(background)
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.style.paint.corner_radii = CornerRadii::all(radius.max(0.0));
        self
    }

    pub fn rounded(self, radius: f32) -> Self {
        self.corner_radius(radius)
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
        self.style.paint.border = Border::all(width, color);
        self
    }

    pub fn border_style(mut self, border: Border) -> Self {
        self.style.paint.border = Border {
            widths: EdgeWidths {
                top: border.widths.top.max(0.0),
                right: border.widths.right.max(0.0),
                bottom: border.widths.bottom.max(0.0),
                left: border.widths.left.max(0.0),
            },
            color: border.color,
        };
        self
    }

    pub fn border_widths(mut self, border_widths: EdgeWidths) -> Self {
        self.style.paint.border.widths = EdgeWidths {
            top: border_widths.top.max(0.0),
            right: border_widths.right.max(0.0),
            bottom: border_widths.bottom.max(0.0),
            left: border_widths.left.max(0.0),
        };
        self
    }

    pub fn border_color(mut self, color: Color) -> Self {
        self.style.paint.border.color = Some(color);
        self
    }

    pub fn overflow(mut self, overflow: Overflow) -> Self {
        self.style.layout.overflow = overflow;
        self
    }

    pub fn overflow_hidden(self) -> Self {
        self.overflow(Overflow::Hidden)
    }

    pub fn overflow_visible(self) -> Self {
        self.overflow(Overflow::Visible)
    }

    pub fn clip(self) -> Self {
        self.overflow_hidden()
    }

    impl_shared_text_style_builders!();
    impl_shared_window_chrome_builders!();

    pub fn focusable(mut self) -> Self {
        self.interaction.focus_policy = FocusPolicy::Keyboard;
        self
    }

    pub fn text_input(mut self, purpose: TextInputPurpose) -> Self {
        self.interaction.focus_policy = FocusPolicy::TextInput;
        self.interaction.text_input = Some(TextInputState {
            ime_allowed: true,
            purpose,
            placeholder: None,
        });
        self.semantics.role = SemanticsRole::TextInput;
        self
    }

    pub fn semantics_role(mut self, role: SemanticsRole) -> Self {
        self.semantics.role = role;
        self
    }

    pub fn semantics_label(mut self, label: impl Into<crate::SharedString>) -> Self {
        self.semantics.label = Some(label.into());
        self
    }

    pub fn semantics_value(mut self, value: impl Into<crate::SharedString>) -> Self {
        self.semantics.value = Some(value.into());
        self
    }

    pub fn semantics_hidden(mut self, hidden: bool) -> Self {
        self.semantics.hidden = hidden;
        self
    }

    pub fn semantics_disabled(mut self, disabled: bool) -> Self {
        self.semantics.disabled = disabled;
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
