use crate::SharedString;
use crate::style::{Point, Px};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputNodeId(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FocusPolicy {
    #[default]
    None,
    Keyboard,
    TextInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextInputState {
    pub ime_allowed: bool,
    pub purpose: TextInputPurpose,
    pub placeholder: Option<SharedString>,
}

impl Default for TextInputState {
    fn default() -> Self {
        Self {
            ime_allowed: false,
            purpose: TextInputPurpose::Normal,
            placeholder: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextInputPurpose {
    #[default]
    Normal,
    Password,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaretRect {
    pub origin: Point<Px>,
    pub size: crate::style::Size<Px>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerButton {
    Primary,
    Secondary,
    Middle,
    Back,
    Forward,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerPhase {
    Down,
    Up,
    Move,
    Leave,
    Wheel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerEvent {
    pub phase: PointerPhase,
    pub position: Point<Px>,
    pub button: Option<PointerButton>,
    pub delta: Option<Point<Px>>,
}
