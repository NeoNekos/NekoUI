use crate::error::NekoResult;
use crate::runtime::Runtime;
use crate::window::{AnyWindowHandle, WindowOptions};

pub struct WindowService<'a> {
    runtime: &'a mut Runtime,
}

impl<'a> WindowService<'a> {
    pub(crate) fn new(runtime: &'a mut Runtime) -> Self {
        Self { runtime }
    }

    pub fn open(&mut self, options: WindowOptions) -> NekoResult<AnyWindowHandle> {
        self.runtime.open_window(options)
    }

    pub fn request_close(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.runtime.request_close_window(handle)
    }

    pub fn close(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.runtime.close_window(handle)
    }

    pub fn validate(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.runtime.validate_window(handle)
    }
}
