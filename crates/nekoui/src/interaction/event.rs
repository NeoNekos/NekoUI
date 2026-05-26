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
