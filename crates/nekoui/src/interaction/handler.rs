use std::rc::Rc;

use crate::app::AppContext;
use crate::error::NekoResult;
use crate::interaction::{ClickEvent, PointerEvent};

pub trait IntoHandlerResult {
    fn into_result(self) -> NekoResult<()>;
}

impl IntoHandlerResult for () {
    fn into_result(self) -> NekoResult<()> {
        Ok(())
    }
}

impl IntoHandlerResult for NekoResult<()> {
    fn into_result(self) -> NekoResult<()> {
        self
    }
}

pub(crate) type PointerHandler =
    Rc<dyn for<'a> Fn(&PointerEvent, &mut AppContext<'a>) -> NekoResult<()>>;
pub(crate) type ClickHandler =
    Rc<dyn for<'a> Fn(&ClickEvent, &mut AppContext<'a>) -> NekoResult<()>>;

#[derive(Clone, Default)]
pub(crate) struct InteractionHandlers {
    pointer_down: Option<PointerHandler>,
    pointer_up: Option<PointerHandler>,
    pointer_move: Option<PointerHandler>,
    click: Option<ClickHandler>,
}

impl std::fmt::Debug for InteractionHandlers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InteractionHandlers")
            .field("pointer_down", &self.pointer_down.is_some())
            .field("pointer_up", &self.pointer_up.is_some())
            .field("pointer_move", &self.pointer_move.is_some())
            .field("click", &self.click.is_some())
            .finish()
    }
}

impl PartialEq for InteractionHandlers {
    fn eq(&self, other: &Self) -> bool {
        self.pointer_down.is_some() == other.pointer_down.is_some()
            && self.pointer_up.is_some() == other.pointer_up.is_some()
            && self.pointer_move.is_some() == other.pointer_move.is_some()
            && self.click.is_some() == other.click.is_some()
    }
}

impl InteractionHandlers {
    pub(crate) fn set_pointer_down(&mut self, handler: PointerHandler) {
        self.pointer_down = Some(handler);
    }

    pub(crate) fn set_pointer_up(&mut self, handler: PointerHandler) {
        self.pointer_up = Some(handler);
    }

    pub(crate) fn set_pointer_move(&mut self, handler: PointerHandler) {
        self.pointer_move = Some(handler);
    }

    pub(crate) fn set_click(&mut self, handler: ClickHandler) {
        self.click = Some(handler);
    }

    pub(crate) fn pointer_down(&self) -> Option<PointerHandler> {
        self.pointer_down.clone()
    }

    pub(crate) fn pointer_up(&self) -> Option<PointerHandler> {
        self.pointer_up.clone()
    }

    pub(crate) fn pointer_move(&self) -> Option<PointerHandler> {
        self.pointer_move.clone()
    }

    pub(crate) fn click(&self) -> Option<ClickHandler> {
        self.click.clone()
    }
}
