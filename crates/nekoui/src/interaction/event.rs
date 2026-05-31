use crate::layout::LayoutPoint;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PointerButton {
    Primary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PointerInputKind {
    Move,
    Down,
    Up,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerInput {
    kind: PointerInputKind,
    position: LayoutPoint,
    button: PointerButton,
    pointer_id: u64,
}

impl PointerInput {
    pub fn move_to(position: LayoutPoint) -> Self {
        Self::new(PointerInputKind::Move, position)
    }

    pub fn down(position: LayoutPoint) -> Self {
        Self::new(PointerInputKind::Down, position)
    }

    pub fn up(position: LayoutPoint) -> Self {
        Self::new(PointerInputKind::Up, position)
    }

    pub fn cancel(position: LayoutPoint) -> Self {
        Self::new(PointerInputKind::Cancel, position)
    }

    pub fn new(kind: PointerInputKind, position: LayoutPoint) -> Self {
        Self {
            kind,
            position,
            button: PointerButton::Primary,
            pointer_id: 1,
        }
    }

    pub fn kind(self) -> PointerInputKind {
        self.kind
    }

    pub fn position(self) -> LayoutPoint {
        self.position
    }

    pub fn button(self) -> PointerButton {
        self.button
    }

    pub fn pointer_id(self) -> u64 {
        self.pointer_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum KeyInputKind {
    Down,
    Up,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Key {
    Character(String),
    Named(String),
    Dead(Option<char>),
    Unidentified,
}

impl Key {
    pub fn character(value: impl Into<String>) -> Self {
        Self::Character(value.into())
    }

    pub fn named(value: impl Into<String>) -> Self {
        Self::Named(value.into())
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Character(value) | Self::Named(value) => value,
            Self::Dead(Some(_)) => "dead",
            Self::Dead(None) => "dead_unknown",
            Self::Unidentified => "unidentified",
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Character(_) => "character",
            Self::Named(_) => "named",
            Self::Dead(_) => "dead",
            Self::Unidentified => "unidentified",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalKey {
    Code(String),
    Unidentified,
}

impl PhysicalKey {
    pub fn code(value: impl Into<String>) -> Self {
        Self::Code(value.into())
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Code(value) => value,
            Self::Unidentified => "unidentified",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Modifiers {
    bits: u8,
}

impl Modifiers {
    const SHIFT: u8 = 1 << 0;
    const CTRL: u8 = 1 << 1;
    const ALT: u8 = 1 << 2;
    const LOGO: u8 = 1 << 3;

    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub const fn new(shift: bool, ctrl: bool, alt: bool, logo: bool) -> Self {
        let mut bits = 0;
        if shift {
            bits |= Self::SHIFT;
        }
        if ctrl {
            bits |= Self::CTRL;
        }
        if alt {
            bits |= Self::ALT;
        }
        if logo {
            bits |= Self::LOGO;
        }
        Self { bits }
    }

    pub const fn bits(self) -> u8 {
        self.bits
    }

    pub const fn shift(self) -> bool {
        self.bits & Self::SHIFT != 0
    }

    pub const fn ctrl(self) -> bool {
        self.bits & Self::CTRL != 0
    }

    pub const fn control(self) -> bool {
        self.ctrl()
    }

    pub const fn alt(self) -> bool {
        self.bits & Self::ALT != 0
    }

    pub const fn logo(self) -> bool {
        self.bits & Self::LOGO != 0
    }

    #[cfg(target_os = "macos")]
    pub const fn command(self) -> bool {
        self.logo()
    }

    #[cfg(not(target_os = "macos"))]
    pub const fn command(self) -> bool {
        self.ctrl()
    }
}

#[cfg(test)]
mod tests {
    use super::Modifiers;

    #[cfg(target_os = "macos")]
    #[test]
    fn command_modifier_uses_logo_on_macos() {
        assert!(Modifiers::new(false, false, false, true).command());
        assert!(!Modifiers::new(false, true, false, false).command());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn command_modifier_uses_ctrl_off_macos() {
        assert!(Modifiers::new(false, true, false, false).command());
        assert!(!Modifiers::new(false, false, false, true).command());
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct KeyInput {
    kind: KeyInputKind,
    logical_key: Key,
    physical_key: PhysicalKey,
    modifiers: Modifiers,
    repeat: bool,
    synthetic: bool,
}

impl KeyInput {
    pub fn down(logical_key: Key) -> Self {
        Self::new(KeyInputKind::Down, logical_key)
    }

    pub fn up(logical_key: Key) -> Self {
        Self::new(KeyInputKind::Up, logical_key)
    }

    pub fn new(kind: KeyInputKind, logical_key: Key) -> Self {
        Self {
            kind,
            logical_key,
            physical_key: PhysicalKey::Unidentified,
            modifiers: Modifiers::empty(),
            repeat: false,
            synthetic: false,
        }
    }

    pub fn with_physical_key(mut self, physical_key: PhysicalKey) -> Self {
        self.physical_key = physical_key;
        self
    }

    pub fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    pub fn with_repeat(mut self, repeat: bool) -> Self {
        self.repeat = repeat;
        self
    }

    pub fn with_synthetic(mut self, synthetic: bool) -> Self {
        self.synthetic = synthetic;
        self
    }

    pub fn kind(&self) -> KeyInputKind {
        self.kind
    }

    pub fn logical_key(&self) -> &Key {
        &self.logical_key
    }

    pub fn physical_key(&self) -> &PhysicalKey {
        &self.physical_key
    }

    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    pub fn repeat(&self) -> bool {
        self.repeat
    }

    pub fn synthetic(&self) -> bool {
        self.synthetic
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct KeyEvent {
    input: KeyInput,
}

impl KeyEvent {
    pub(crate) fn new(input: KeyInput) -> Self {
        Self { input }
    }

    pub fn input(&self) -> &KeyInput {
        &self.input
    }

    pub fn kind(&self) -> KeyInputKind {
        self.input.kind()
    }

    pub fn logical_key(&self) -> &Key {
        self.input.logical_key()
    }

    pub fn physical_key(&self) -> &PhysicalKey {
        self.input.physical_key()
    }

    pub fn modifiers(&self) -> Modifiers {
        self.input.modifiers()
    }

    pub fn repeat(&self) -> bool {
        self.input.repeat()
    }

    pub fn synthetic(&self) -> bool {
        self.input.synthetic()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScrollDelta {
    Lines { x: f32, y: f32 },
    Pixels { x: f32, y: f32 },
}

impl ScrollDelta {
    pub const fn lines(x: f32, y: f32) -> Self {
        Self::Lines { x, y }
    }

    pub const fn pixels(x: f32, y: f32) -> Self {
        Self::Pixels { x, y }
    }

    pub const fn x(self) -> f32 {
        match self {
            Self::Lines { x, .. } | Self::Pixels { x, .. } => x,
        }
    }

    pub const fn y(self) -> f32 {
        match self {
            Self::Lines { y, .. } | Self::Pixels { y, .. } => y,
        }
    }

    pub const fn unit_name(self) -> &'static str {
        match self {
            Self::Lines { .. } => "lines",
            Self::Pixels { .. } => "pixels",
        }
    }

    pub fn is_finite(self) -> bool {
        self.x().is_finite() && self.y().is_finite()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScrollPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
    Unknown,
}

impl ScrollPhase {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Moved => "moved",
            Self::Ended => "ended",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelInput {
    delta: ScrollDelta,
    phase: ScrollPhase,
    modifiers: Modifiers,
}

impl WheelInput {
    pub fn new(delta: ScrollDelta, phase: ScrollPhase) -> Self {
        Self {
            delta,
            phase,
            modifiers: Modifiers::empty(),
        }
    }

    pub fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    pub fn delta(self) -> ScrollDelta {
        self.delta
    }

    pub fn phase(self) -> ScrollPhase {
        self.phase
    }

    pub fn modifiers(self) -> Modifiers {
        self.modifiers
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TextRange {
    start: usize,
    end: usize,
}

impl TextRange {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn collapsed(offset: usize) -> Self {
        Self {
            start: offset,
            end: offset,
        }
    }

    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn end(self) -> usize {
        self.end
    }

    pub const fn is_collapsed(self) -> bool {
        self.start == self.end
    }

    pub(crate) fn validate_for_text(self, text: &str) -> bool {
        self.start <= self.end
            && self.end <= text.len()
            && text.is_char_boundary(self.start)
            && text.is_char_boundary(self.end)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TextInput {
    text: String,
    replace: Option<TextRange>,
}

impl TextInput {
    pub fn commit(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            replace: None,
        }
    }

    pub fn with_replace(mut self, range: TextRange) -> Self {
        self.replace = Some(range);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn replace(&self) -> Option<TextRange> {
        self.replace
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ImePreeditInput {
    text: String,
    cursor: Option<TextRange>,
    replace: Option<TextRange>,
}

impl ImePreeditInput {
    pub fn new(text: impl Into<String>, cursor: Option<TextRange>) -> Self {
        Self {
            text: text.into(),
            cursor,
            replace: None,
        }
    }

    pub fn with_replace(mut self, range: TextRange) -> Self {
        self.replace = Some(range);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> Option<TextRange> {
        self.cursor
    }

    pub fn replace(&self) -> Option<TextRange> {
        self.replace
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ImeInput {
    Enabled,
    Preedit(ImePreeditInput),
    Commit(TextInput),
    Disabled,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum TextInputPurpose {
    #[default]
    Normal,
    Password,
    Terminal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WindowFocusInput {
    focused: bool,
}

impl WindowFocusInput {
    pub const fn new(focused: bool) -> Self {
        Self { focused }
    }

    pub const fn focused(self) -> bool {
        self.focused
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerEvent {
    input: PointerInput,
}

impl PointerEvent {
    pub(crate) fn new(input: PointerInput) -> Self {
        Self { input }
    }

    pub fn input(self) -> PointerInput {
        self.input
    }

    pub fn position(self) -> LayoutPoint {
        self.input.position()
    }

    pub fn button(self) -> PointerButton {
        self.input.button()
    }

    pub fn pointer_id(self) -> u64 {
        self.input.pointer_id()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClickEvent {
    pointer: PointerEvent,
}

impl ClickEvent {
    pub(crate) fn new(pointer: PointerEvent) -> Self {
        Self { pointer }
    }

    pub fn pointer(self) -> PointerEvent {
        self.pointer
    }

    pub fn position(self) -> LayoutPoint {
        self.pointer.position()
    }
}
