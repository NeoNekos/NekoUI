use std::borrow::Cow;

use crate::error::{NekoError, NekoResult};
use crate::window::{AnyWindowHandle, WindowOptions};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WindowLifecycle {
    Live,
    CloseRequested,
    Closed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowRecord {
    handle: AnyWindowHandle,
    title: Cow<'static, str>,
    lifecycle: WindowLifecycle,
}

impl WindowRecord {
    pub(crate) fn new(handle: AnyWindowHandle, options: WindowOptions) -> Self {
        Self {
            handle,
            title: options.title,
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
