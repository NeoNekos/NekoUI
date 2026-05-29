use std::borrow::Cow;

use crate::error::{NekoError, NekoResult};
use crate::layout::{LayoutSize, Viewport};
use crate::platform::{PhysicalSize, Renderability};
use crate::window::{AnyWindowHandle, WindowOptions};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WindowLifecycle {
    Requested,
    CreatedHidden,
    Visible,
    Hidden,
    Minimized,
    ClosePending,
    Closing,
    Destroyed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WindowRecord {
    handle: AnyWindowHandle,
    title: Cow<'static, str>,
    viewport: Viewport,
    lifecycle: WindowLifecycle,
    native_created: bool,
    physical_size: PhysicalSize,
    renderability: Renderability,
    surface_generation: u64,
}

impl WindowRecord {
    pub(crate) fn new(handle: AnyWindowHandle, options: WindowOptions) -> Self {
        Self {
            handle,
            title: options.title,
            viewport: options.viewport,
            lifecycle: WindowLifecycle::Requested,
            native_created: false,
            physical_size: PhysicalSize::ZERO,
            renderability: Renderability::SurfaceAbsent,
            surface_generation: 1,
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

    pub(crate) fn native_created(&self) -> bool {
        self.native_created
    }

    pub(crate) fn renderability(&self) -> Renderability {
        self.renderability
    }

    pub(crate) fn physical_size(&self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) fn surface_generation(&self) -> u64 {
        self.surface_generation
    }

    pub(crate) fn resize(&mut self, logical_size: LayoutSize) {
        self.viewport = self
            .viewport
            .next_generation(logical_size, self.viewport.scale_factor());
        self.bump_surface_generation();
    }

    pub(crate) fn rescale(&mut self, scale_factor: f32) -> NekoResult<bool> {
        if !(scale_factor.is_finite() && scale_factor > 0.0) {
            return Err(NekoError::invalid_input(
                "viewport scale factor must be finite and positive",
            ));
        }
        if self.viewport.scale_factor() == scale_factor {
            return Ok(false);
        }
        self.viewport = self
            .viewport
            .next_generation(self.viewport.logical_size(), scale_factor);
        self.bump_surface_generation();
        Ok(true)
    }

    pub(crate) fn mark_native_created(&mut self) {
        self.native_created = true;
        if self.lifecycle == WindowLifecycle::Requested {
            self.lifecycle = WindowLifecycle::CreatedHidden;
        }
    }

    pub(crate) fn show(&mut self) {
        if matches!(
            self.lifecycle,
            WindowLifecycle::Requested | WindowLifecycle::CreatedHidden | WindowLifecycle::Hidden
        ) {
            self.lifecycle = WindowLifecycle::Visible;
        }
    }

    pub(crate) fn minimize(&mut self) {
        if matches!(
            self.lifecycle,
            WindowLifecycle::Visible | WindowLifecycle::Hidden
        ) {
            self.lifecycle = WindowLifecycle::Minimized;
        }
        self.set_renderability(Renderability::Minimized);
    }

    pub(crate) fn restore(&mut self) {
        if self.lifecycle == WindowLifecycle::Minimized {
            self.lifecycle = WindowLifecycle::Visible;
        }
        if matches!(
            self.renderability,
            Renderability::Closing | Renderability::Destroyed
        ) {
            return;
        }
        if self.physical_size.is_zero() {
            self.set_renderability(Renderability::ZeroSize);
        } else {
            self.set_renderability(Renderability::Renderable);
        }
    }

    pub(crate) fn request_close(&mut self) {
        if !matches!(
            self.lifecycle,
            WindowLifecycle::Closing | WindowLifecycle::Destroyed
        ) {
            self.lifecycle = WindowLifecycle::ClosePending;
        }
    }

    pub(crate) fn confirm_close(&mut self) {
        if self.lifecycle == WindowLifecycle::ClosePending {
            self.lifecycle = WindowLifecycle::Closing;
        }
        self.set_renderability(Renderability::Closing);
    }

    pub(crate) fn close(&mut self) {
        self.lifecycle = WindowLifecycle::Destroyed;
        self.set_renderability(Renderability::Destroyed);
    }

    pub(crate) fn set_physical_size(&mut self, physical_size: PhysicalSize) -> bool {
        if self.physical_size == physical_size {
            return false;
        }
        self.physical_size = physical_size;
        self.bump_surface_generation();
        true
    }

    pub(crate) fn set_renderability(&mut self, renderability: Renderability) -> bool {
        if self.renderability == renderability {
            return false;
        }
        self.renderability = renderability;
        if !renderability.is_renderable() || self.physical_size != PhysicalSize::ZERO {
            self.bump_surface_generation();
        }
        true
    }

    pub(crate) fn ensure_live(&self, handle: AnyWindowHandle) -> NekoResult<()> {
        if self.handle != handle || self.lifecycle == WindowLifecycle::Destroyed {
            Err(NekoError::stale("window handle is stale"))
        } else {
            Ok(())
        }
    }

    fn bump_surface_generation(&mut self) {
        self.surface_generation = self.surface_generation.saturating_add(1);
    }
}

impl WindowLifecycle {
    pub fn name(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::CreatedHidden => "created_hidden",
            Self::Visible => "visible",
            Self::Hidden => "hidden",
            Self::Minimized => "minimized",
            Self::ClosePending => "close_pending",
            Self::Closing => "closing",
            Self::Destroyed => "destroyed",
        }
    }
}
