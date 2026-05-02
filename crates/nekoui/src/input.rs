use crate::SharedString;
use crate::style::{Point, Px};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputNodeId(pub(crate) u64);

impl InputNodeId {
    pub(crate) fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_INPUT_NODE_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_INPUT_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FocusPolicy {
    #[default]
    None,
    Keyboard,
    TextInput,
}

#[derive(Debug, Clone)]
pub struct TextInputState {
    pub ime_allowed: bool,
    pub purpose: TextInputPurpose,
    pub placeholder: Option<SharedString>,
}

impl PartialEq for TextInputState {
    fn eq(&self, other: &Self) -> bool {
        self.ime_allowed == other.ime_allowed
            && self.purpose == other.purpose
            && self.placeholder == other.placeholder
    }
}

impl Eq for TextInputState {}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextInputEvent {
    Commit {
        source: u64,
        target: InputNodeId,
        text: SharedString,
    },
    Preedit {
        source: u64,
        target: InputNodeId,
        text: Option<SharedString>,
    },
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
