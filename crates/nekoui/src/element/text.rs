use crate::SharedString;
use crate::input::{FocusPolicy, TextInputPurpose, TextInputState};
use crate::semantics::{SemanticsRole, SemanticsState};
use crate::style::{
    Absolute, AlignItems, AlignSelf, BoxSizing, Color, Definite, Edges, FontStyle, FontWeight,
    IntoFontFamilies, LayoutSize, Length, Size, Style, TextAlign, TextOverflow, WhiteSpace,
};

use super::builder_macros::{
    impl_shared_flex_item_builders, impl_shared_key_size_margin_builders,
    impl_shared_text_style_builders, impl_shared_window_chrome_builders,
};
use super::{AnyElement, InteractionState, IntoElement, WindowFrameArea};

#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub(crate) key: Option<u64>,
    pub(crate) style: Style,
    pub(crate) window_frame_area: Option<WindowFrameArea>,
    pub(crate) interaction: InteractionState,
    pub(crate) semantics: SemanticsState,
    pub(crate) content: SharedString,
}

pub fn text(content: impl Into<SharedString>) -> Text {
    Text {
        key: None,
        style: Style::default(),
        window_frame_area: None,
        interaction: InteractionState::default(),
        semantics: SemanticsState {
            role: SemanticsRole::Text,
            ..SemanticsState::default()
        },
        content: content.into(),
    }
}

impl Text {
    impl_shared_key_size_margin_builders!();
    impl_shared_flex_item_builders!();
    impl_shared_text_style_builders!();

    pub fn text_overflow(mut self, overflow: TextOverflow) -> Self {
        self.style.text.text_overflow = Some(overflow);
        self
    }

    pub fn text_ellipsis(self) -> Self {
        self.text_overflow(TextOverflow::Ellipsis)
    }

    pub fn text_clip(self) -> Self {
        self.text_overflow(TextOverflow::Clip)
    }

    pub fn truncate(self) -> Self {
        self.whitespace_nowrap().text_ellipsis()
    }

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

impl IntoElement for Text {
    fn into_any_element(self) -> AnyElement {
        AnyElement::text(self)
    }
}
