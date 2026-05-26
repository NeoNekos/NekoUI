use std::borrow::Cow;

use crate::error::{NekoError, NekoResult};
use crate::layout::{LayoutSize, Viewport};
use crate::window::{AnyWindowHandle, WindowOptions};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WindowLifecycle {
    Live,
    CloseRequested,
    Closed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WindowRecord {
    handle: AnyWindowHandle,
    title: Cow<'static, str>,
    viewport: Viewport,
    lifecycle: WindowLifecycle,
}

impl WindowRecord {
    pub(crate) fn new(handle: AnyWindowHandle, options: WindowOptions) -> Self {
        Self {
            handle,
            title: options.title,
            viewport: options.viewport,
            lifecycle: WindowLifecycle::Live,
        }
    }

    pub fn handle(&self) -> AnyWindowHandle {
        self.handle
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn lifecycle(&self) -> WindowLifecycle {
        self.lifecycle
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    pub(crate) fn resize(&mut self, logical_size: LayoutSize) {
        self.viewport = self
            .viewport
            .next_generation(logical_size, self.viewport.scale_factor());
    }

    pub(crate) fn request_close(&mut self) {
        if self.lifecycle == WindowLifecycle::Live {
            self.lifecycle = WindowLifecycle::CloseRequested;
        }
    }

    pub(crate) fn close(&mut self) {
        self.lifecycle = WindowLifecycle::Closed;
    }

    pub(crate) fn ensure_live(&self, handle: AnyWindowHandle) -> NekoResult<()> {
        if self.handle != handle || self.lifecycle == WindowLifecycle::Closed {
            Err(NekoError::stale("window handle is stale"))
        } else {
            Ok(())
        }
    }
}
