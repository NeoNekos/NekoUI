mod contract;
#[cfg(target_os = "windows")]
mod windows;
mod winit_runtime;

pub(crate) use contract::PlatformFact;
#[cfg(target_os = "windows")]
pub(crate) use contract::{BackendFrameReceipt, BackendFrameStatus, BackendSurfaceState};
pub(crate) use contract::{PhysicalSize, Renderability};
#[cfg(target_os = "windows")]
pub(crate) use windows::NativeRenderer;
pub(crate) use winit_runtime::ApplicationPlatform;
