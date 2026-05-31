#[cfg(target_os = "windows")]
use crate::error::ErrorKind;
use crate::interaction::{
    ImeInput, KeyInput, Modifiers, PointerInput, TextInput, WheelInput, WindowFocusInput,
};
use crate::layout::LayoutSize;
use crate::window::AnyWindowHandle;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PlatformFact {
    WindowCreated {
        handle: AnyWindowHandle,
    },
    WindowShown {
        handle: AnyWindowHandle,
    },
    CloseRequested {
        handle: AnyWindowHandle,
    },
    CloseConfirmed {
        handle: AnyWindowHandle,
    },
    Destroyed {
        handle: AnyWindowHandle,
    },
    LogicalSizeChanged {
        handle: AnyWindowHandle,
        logical_size: LayoutSize,
    },
    PhysicalSizeChanged {
        handle: AnyWindowHandle,
        physical_size: PhysicalSize,
    },
    ScaleFactorChanged {
        handle: AnyWindowHandle,
        scale_factor: f32,
    },
    Minimized {
        handle: AnyWindowHandle,
    },
    Restored {
        handle: AnyWindowHandle,
    },
    RenderabilityChanged {
        handle: AnyWindowHandle,
        renderability: Renderability,
    },
    RedrawRequested {
        handle: AnyWindowHandle,
    },
    Wake,
    Exit,
    PointerInput {
        handle: AnyWindowHandle,
        input: PointerInput,
    },
    KeyInput {
        handle: AnyWindowHandle,
        input: KeyInput,
    },
    TextInput {
        handle: AnyWindowHandle,
        input: TextInput,
    },
    ImeInput {
        handle: AnyWindowHandle,
        input: ImeInput,
    },
    ModifiersChanged {
        handle: AnyWindowHandle,
        modifiers: Modifiers,
    },
    WheelInput {
        handle: AnyWindowHandle,
        input: WheelInput,
    },
    WindowFocusChanged {
        handle: AnyWindowHandle,
        input: WindowFocusInput,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ImePlatformRequest {
    Allowed {
        allowed: bool,
    },
    CursorArea {
        rect: crate::layout::LayoutRect,
    },
    Purpose {
        purpose: crate::interaction::TextInputPurpose,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalSize {
    width: u32,
    height: u32,
}

impl PhysicalSize {
    pub(crate) const ZERO: Self = Self {
        width: 0,
        height: 0,
    };

    pub(crate) const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub(crate) const fn is_zero(self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[cfg(target_os = "windows")]
    pub(crate) const fn width(self) -> u32 {
        self.width
    }

    #[cfg(target_os = "windows")]
    pub(crate) const fn height(self) -> u32 {
        self.height
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Renderability {
    #[default]
    Renderable,
    ZeroSize,
    Minimized,
    SurfaceAbsent,
    Closing,
    Destroyed,
}

impl Renderability {
    pub(crate) const fn is_renderable(self) -> bool {
        matches!(self, Self::Renderable)
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Renderable => "renderable",
            Self::ZeroSize => "zero_size",
            Self::Minimized => "minimized",
            Self::SurfaceAbsent => "surface_absent",
            Self::Closing => "closing",
            Self::Destroyed => "destroyed",
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendSurfaceState {
    Absent,
    Ready,
    Suspended,
    Lost,
    Destroyed,
}

#[cfg(target_os = "windows")]
impl BackendSurfaceState {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Ready => "ready",
            Self::Suspended => "suspended",
            Self::Lost => "lost",
            Self::Destroyed => "destroyed",
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendFrameStatus {
    Presented,
    NotRenderable,
    StaleDropped,
    Aborted,
    Failed,
}

#[cfg(target_os = "windows")]
impl BackendFrameStatus {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Presented => "presented",
            Self::NotRenderable => "not_renderable",
            Self::StaleDropped => "stale_dropped",
            Self::Aborted => "aborted",
            Self::Failed => "failed",
        }
    }

    pub(crate) const fn diagnostic_category(self, failure_kind: Option<ErrorKind>) -> ErrorKind {
        match self {
            Self::Presented => ErrorKind::Diagnostic,
            Self::NotRenderable => ErrorKind::NotRenderable,
            Self::StaleDropped => ErrorKind::Stale,
            Self::Aborted => ErrorKind::Diagnostic,
            Self::Failed => match failure_kind {
                Some(kind) => kind,
                None => ErrorKind::Diagnostic,
            },
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BackendFrameReceipt {
    pub(crate) backend: &'static str,
    pub(crate) window: AnyWindowHandle,
    pub(crate) surface_generation: u64,
    pub(crate) status: BackendFrameStatus,
    pub(crate) failure_kind: Option<ErrorKind>,
    pub(crate) surface_state: BackendSurfaceState,
    pub(crate) unsupported_draw_items: usize,
    pub(crate) stale_drop_count: u64,
    pub(crate) message: &'static str,
}

#[cfg(target_os = "windows")]
impl BackendFrameReceipt {
    pub(crate) const fn diagnostic_category(&self) -> ErrorKind {
        self.status.diagnostic_category(self.failure_kind)
    }
}
