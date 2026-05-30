use crate::app::{Context, Entity, Render};
use crate::error::NekoResult;
use crate::interaction::PointerInput;
#[cfg(test)]
use crate::interaction::{KeyInput, WheelInput, WindowFocusInput};
use crate::layout::{LayoutPoint, LayoutSize};
use crate::retained::RetainedTreeSnapshot;
use crate::runtime::Runtime;
use crate::scene::PaintScene;
use crate::window::{AnyWindowHandle, WindowHandle, WindowOptions};

pub struct WindowService<'a> {
    runtime: &'a mut Runtime,
}

impl<'a> WindowService<'a> {
    pub(crate) fn new(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
    }

    pub fn open<T: Render>(
        &mut self,
        options: WindowOptions,
        build: impl FnOnce(&mut Context<'_, T>) -> T,
    ) -> NekoResult<WindowHandle<T>> {
        self.runtime.open_window(options, build)
    }

    pub fn root_view<T: Render>(&mut self, handle: WindowHandle<T>) -> NekoResult<Entity<T>> {
        self.runtime.root_view(handle)
    }

    pub fn request_close(&mut self, handle: impl Into<AnyWindowHandle>) -> NekoResult<()> {
        self.runtime.request_close_window(handle)
    }

    pub fn close(&mut self, handle: impl Into<AnyWindowHandle>) -> NekoResult<()> {
        self.runtime.close_window(handle)
    }

    pub fn resize(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        logical_size: LayoutSize,
    ) -> NekoResult<()> {
        self.runtime.resize_window(handle, logical_size)
    }

    pub fn pointer_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: PointerInput,
    ) -> NekoResult<()> {
        self.runtime.pointer_input(handle, input)
    }

    pub fn pointer_move(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        position: crate::layout::LayoutPoint,
    ) -> NekoResult<()> {
        self.pointer_input(handle, PointerInput::move_to(position))
    }

    pub fn pointer_down(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        position: crate::layout::LayoutPoint,
    ) -> NekoResult<()> {
        self.pointer_input(handle, PointerInput::down(position))
    }

    pub fn pointer_up(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        position: crate::layout::LayoutPoint,
    ) -> NekoResult<()> {
        self.pointer_input(handle, PointerInput::up(position))
    }

    pub fn pointer_cancel(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        position: LayoutPoint,
    ) -> NekoResult<()> {
        self.pointer_input(handle, PointerInput::cancel(position))
    }

    #[cfg(test)]
    pub(crate) fn wheel_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: WheelInput,
    ) -> NekoResult<()> {
        self.runtime.wheel_input(handle, input)
    }

    #[cfg(test)]
    pub(crate) fn key_input(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: KeyInput,
    ) -> NekoResult<()> {
        self.runtime.key_input(handle, input)
    }

    #[cfg(test)]
    pub(crate) fn window_focus_changed(
        &mut self,
        handle: impl Into<AnyWindowHandle>,
        input: WindowFocusInput,
    ) -> NekoResult<()> {
        self.runtime.window_focus_changed(handle, input)
    }

    pub fn validate(&mut self, handle: impl Into<AnyWindowHandle>) -> NekoResult<()> {
        self.runtime.validate_window(handle)
    }

    pub fn retained_snapshot(
        &self,
        handle: impl Into<AnyWindowHandle>,
    ) -> NekoResult<RetainedTreeSnapshot> {
        self.runtime.retained_snapshot(handle)
    }

    pub fn scene_snapshot(&self, handle: impl Into<AnyWindowHandle>) -> NekoResult<PaintScene> {
        self.runtime.scene_snapshot(handle)
    }
}
