use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::Win32::Foundation::HWND;
use winit::window::Window;

use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeWindowHandle {
    hwnd: HWND,
}

impl NativeWindowHandle {
    pub(super) fn hwnd(self) -> HWND {
        self.hwnd
    }
}

pub(super) fn hwnd_from_winit(window: &Window) -> NekoResult<NativeWindowHandle> {
    // The returned HWND is a borrowed native fact from `window`; callers must use it only while
    // the source winit Window is still owned by the platform runtime.
    let handle = window
        .window_handle()
        .map_err(|error| NekoError::unavailable(format!("window handle unavailable: {error}")))?;
    match handle.as_raw() {
        RawWindowHandle::Win32(handle) => Ok(NativeWindowHandle {
            hwnd: HWND(handle.hwnd.get() as *mut core::ffi::c_void),
        }),
        _ => Err(NekoError::unavailable("winit window is not a Win32 HWND")),
    }
}
