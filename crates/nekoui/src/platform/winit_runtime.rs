use std::collections::HashMap;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize as WinitPhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key as WinitKey, ModifiersState, PhysicalKey as WinitPhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId as WinitWindowId};

use crate::app::{AppContext, Application};
use crate::error::{NekoError, NekoResult};
use crate::interaction::{
    Key, KeyInput, KeyInputKind, Modifiers, PhysicalKey, PointerInput, ScrollDelta, ScrollPhase,
    WheelInput, WindowFocusInput,
};
use crate::layout::{LayoutPoint, LayoutSize};
#[cfg(target_os = "windows")]
use crate::platform::NativeRenderer;
use crate::platform::{PhysicalSize, PlatformFact, Renderability};
use crate::runtime::Runtime;
use crate::window::{AnyWindowHandle, WindowId, WindowRecord};

pub(crate) struct ApplicationPlatform;

impl ApplicationPlatform {
    pub(crate) fn run(
        _application: Application,
        run: impl FnOnce(&mut AppContext<'_>) -> NekoResult<()>,
    ) -> NekoResult<()> {
        let event_loop = EventLoop::new()
            .map_err(|error| NekoError::unavailable(format!("event loop unavailable: {error}")))?;
        event_loop.set_control_flow(ControlFlow::Wait);

        let mut runtime = Runtime::new();
        {
            let mut cx = AppContext::new(&mut runtime);
            run(&mut cx)?;
        }
        runtime.drain_all()?;

        let mut app = WinitRuntimeApp::new(runtime);
        event_loop
            .run_app(&mut app)
            .map_err(|error| NekoError::diagnostic(format!("event loop failed: {error}")))?;
        Ok(())
    }
}

struct WinitRuntimeApp {
    runtime: Runtime,
    windows_by_neko: HashMap<WindowId, WinitWindowId>,
    handles_by_winit: HashMap<WinitWindowId, AnyWindowHandle>,
    windows: HashMap<WinitWindowId, Window>,
    cursor_positions: HashMap<WinitWindowId, LayoutPoint>,
    modifiers: HashMap<WinitWindowId, Modifiers>,
    #[cfg(target_os = "windows")]
    renderer: Option<NativeRenderer>,
}

impl WinitRuntimeApp {
    fn new(runtime: Runtime) -> Self {
        Self {
            runtime,
            windows_by_neko: HashMap::new(),
            handles_by_winit: HashMap::new(),
            windows: HashMap::new(),
            cursor_positions: HashMap::new(),
            modifiers: HashMap::new(),
            #[cfg(target_os = "windows")]
            renderer: None,
        }
    }

    fn materialize_pending_windows(&mut self, event_loop: &ActiveEventLoop) -> NekoResult<()> {
        for record in self.runtime.windows_needing_native_creation() {
            self.create_native_window(event_loop, &record)?;
        }
        Ok(())
    }

    fn create_native_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        record: &WindowRecord,
    ) -> NekoResult<()> {
        let viewport = record.viewport();
        let logical_size = viewport.logical_size();
        let attributes = WindowAttributes::default()
            .with_title(record.title())
            .with_inner_size(LogicalSize::new(
                f64::from(logical_size.width()),
                f64::from(logical_size.height()),
            ));
        let window = event_loop.create_window(attributes).map_err(|error| {
            NekoError::unavailable(format!("native window unavailable: {error}"))
        })?;
        let winit_id = window.id();
        let handle = record.handle();

        self.windows_by_neko.insert(handle.id(), winit_id);
        self.handles_by_winit.insert(winit_id, handle);
        self.windows.insert(winit_id, window);
        self.modifiers.insert(winit_id, Modifiers::empty());

        self.ingest(PlatformFact::WindowCreated { handle })?;
        self.ingest(PlatformFact::WindowShown { handle })?;
        let Some((physical, scale_factor)) = self.windows.get(&winit_id).map(|window| {
            let physical = window.inner_size();
            let scale_factor = window.scale_factor() as f32;
            (physical, scale_factor)
        }) else {
            return Ok(());
        };
        self.ingest_physical_size(handle, physical)?;
        self.ingest(PlatformFact::ScaleFactorChanged {
            handle,
            scale_factor,
        })?;
        #[cfg(target_os = "windows")]
        self.register_backend_surface(winit_id, handle)?;
        self.forward_platform_requests()?;
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn ensure_renderer(&mut self) -> NekoResult<()> {
        if self.renderer.is_none() {
            self.renderer = Some(NativeRenderer::new(self.runtime.diagnostics_mut())?);
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn register_backend_surface(
        &mut self,
        winit_id: WinitWindowId,
        handle: AnyWindowHandle,
    ) -> NekoResult<()> {
        let record = self.runtime.window_record_for_platform(handle)?;
        let physical_size = record.physical_size();
        let generation = record.surface_generation();
        let renderability = record.renderability();
        self.ensure_renderer()?;
        let mut renderer = self
            .renderer
            .take()
            .ok_or_else(|| NekoError::diagnostic("Windows renderer was not initialized"))?;
        let result = if let Some(window) = self.windows.get(&winit_id) {
            renderer.register_window(
                window,
                handle,
                physical_size,
                generation,
                renderability,
                self.runtime.diagnostics_mut(),
            )
        } else {
            Ok(())
        };
        self.renderer = Some(renderer);
        result
    }

    #[cfg(target_os = "windows")]
    fn update_backend_surface(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let record = self.runtime.window_record_for_platform(handle)?;
        self.ensure_renderer()?;
        let mut renderer = self
            .renderer
            .take()
            .ok_or_else(|| NekoError::diagnostic("Windows renderer was not initialized"))?;
        let winit_id = self.windows_by_neko.get(&handle.id()).copied();
        let window = winit_id.and_then(|winit_id| self.windows.get(&winit_id));
        let result = renderer.resize_or_suspend(
            window,
            handle,
            record.physical_size(),
            record.surface_generation(),
            record.renderability(),
            self.runtime.diagnostics_mut(),
        );
        self.renderer = Some(renderer);
        result
    }

    #[cfg(target_os = "windows")]
    fn destroy_backend_surface(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        if let Some(mut renderer) = self.renderer.take() {
            renderer.destroy_window(handle, self.runtime.diagnostics_mut());
            self.renderer = Some(renderer);
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn render_backend_frame(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let mut renderer = self.renderer.take().ok_or_else(|| {
            NekoError::unavailable("Windows D3D11/DXGI renderer was not initialized")
        })?;
        let result = self
            .runtime
            .render_prepared_frame_for_platform(&mut renderer, handle);
        self.renderer = Some(renderer);
        result.map(|_| ())
    }

    fn ingest(&mut self, fact: PlatformFact) -> NekoResult<()> {
        self.runtime.ingest_platform_fact(fact)?;
        Ok(())
    }

    fn ingest_physical_size(
        &mut self,
        handle: AnyWindowHandle,
        size: WinitPhysicalSize<u32>,
    ) -> NekoResult<()> {
        self.ingest(PlatformFact::PhysicalSizeChanged {
            handle,
            physical_size: PhysicalSize::new(size.width, size.height),
        })?;
        if let Some(winit_id) = self.windows_by_neko.get(&handle.id()).copied()
            && let Some(window) = self.windows.get(&winit_id)
        {
            let scale = window.scale_factor();
            if scale.is_finite() && scale > 0.0 {
                let logical = size.to_logical::<f32>(scale);
                self.ingest(PlatformFact::LogicalSizeChanged {
                    handle,
                    logical_size: LayoutSize::new(logical.width, logical.height),
                })?;
            }
        }
        Ok(())
    }

    fn forward_platform_requests(&mut self) -> NekoResult<()> {
        for handle in self.runtime.take_platform_close_requests() {
            self.remove_native_window(handle)?;
        }
        for handle in self.runtime.take_platform_redraw_requests() {
            if let Some(winit_id) = self.windows_by_neko.get(&handle.id())
                && let Some(window) = self.windows.get(winit_id)
            {
                window.request_redraw();
            }
        }
        Ok(())
    }

    fn drop_native_window(&mut self, handle: AnyWindowHandle) {
        if let Some(winit_id) = self.windows_by_neko.remove(&handle.id()) {
            #[cfg(target_os = "windows")]
            let _ = self.destroy_backend_surface(handle);
            self.handles_by_winit.remove(&winit_id);
            self.windows.remove(&winit_id);
            self.cursor_positions.remove(&winit_id);
            self.modifiers.remove(&winit_id);
        }
    }
    fn remove_native_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.drop_native_window(handle);
        self.ingest(PlatformFact::Destroyed { handle })
    }

    fn ingest_pointer_move(
        &mut self,
        window_id: WinitWindowId,
        handle: AnyWindowHandle,
        position: winit::dpi::PhysicalPosition<f64>,
    ) -> NekoResult<()> {
        let Some(window) = self.windows.get(&window_id) else {
            return Ok(());
        };
        let scale = window.scale_factor();
        if !(scale.is_finite() && scale > 0.0) {
            return Ok(());
        }
        let position = logical_cursor_position(position, scale);
        self.cursor_positions.insert(window_id, position);
        self.ingest(PlatformFact::PointerInput {
            handle,
            input: PointerInput::move_to(position),
        })
    }

    fn ingest_mouse_button(
        &mut self,
        handle: AnyWindowHandle,
        state: ElementState,
        button: MouseButton,
    ) -> NekoResult<()> {
        if button != MouseButton::Left {
            return Ok(());
        }
        let Some(winit_id) = self.windows_by_neko.get(&handle.id()).copied() else {
            return Ok(());
        };
        let position = self
            .cursor_positions
            .get(&winit_id)
            .copied()
            .unwrap_or_else(|| LayoutPoint::new(0.0, 0.0));
        let input = match state {
            ElementState::Pressed => PointerInput::down(position),
            ElementState::Released => PointerInput::up(position),
        };
        self.ingest(PlatformFact::PointerInput { handle, input })
    }

    fn ingest_keyboard_input(
        &mut self,
        window_id: WinitWindowId,
        handle: AnyWindowHandle,
        event: winit::event::KeyEvent,
        synthetic: bool,
    ) -> NekoResult<()> {
        let modifiers = self.modifiers.get(&window_id).copied().unwrap_or_default();
        let kind = match event.state {
            ElementState::Pressed => KeyInputKind::Down,
            ElementState::Released => KeyInputKind::Up,
        };
        let input = KeyInput::new(kind, normalize_key(event.logical_key))
            .with_physical_key(normalize_physical_key(event.physical_key))
            .with_modifiers(modifiers)
            .with_repeat(event.repeat)
            .with_synthetic(synthetic);

        self.ingest(PlatformFact::KeyInput { handle, input })
    }

    fn ingest_modifiers_changed(
        &mut self,
        window_id: WinitWindowId,
        handle: AnyWindowHandle,
        state: ModifiersState,
    ) -> NekoResult<()> {
        let modifiers = normalize_modifiers(state);
        self.modifiers.insert(window_id, modifiers);
        self.ingest(PlatformFact::ModifiersChanged { handle, modifiers })
    }

    fn ingest_mouse_wheel(
        &mut self,
        window_id: WinitWindowId,
        handle: AnyWindowHandle,
        delta: MouseScrollDelta,
        phase: TouchPhase,
    ) -> NekoResult<()> {
        let modifiers = self.modifiers.get(&window_id).copied().unwrap_or_default();
        let delta = match delta {
            MouseScrollDelta::LineDelta(x, y) => logical_scroll_lines(x, y),
            MouseScrollDelta::PixelDelta(position) => {
                let Some(window) = self.windows.get(&window_id) else {
                    return Ok(());
                };
                let scale = window.scale_factor();
                if !(scale.is_finite() && scale > 0.0) {
                    return Ok(());
                }
                logical_scroll_pixels(position, scale)
            }
        };
        self.ingest(PlatformFact::WheelInput {
            handle,
            input: WheelInput::new(delta, normalize_scroll_phase(phase)).with_modifiers(modifiers),
        })
    }

    fn ingest_window_focus(&mut self, handle: AnyWindowHandle, focused: bool) -> NekoResult<()> {
        self.ingest(PlatformFact::WindowFocusChanged {
            handle,
            input: WindowFocusInput::new(focused),
        })
    }

    fn exit_event_loop(&mut self, event_loop: &ActiveEventLoop) {
        let _ = self.ingest(PlatformFact::Exit);
        event_loop.exit();
    }

    fn should_exit(&self) -> bool {
        self.runtime.live_windows_for_platform().is_empty() || self.windows.is_empty()
    }
}

impl ApplicationHandler for WinitRuntimeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ingest(PlatformFact::Wake).is_err()
            || self.materialize_pending_windows(event_loop).is_err()
            || self.should_exit()
        {
            self.exit_event_loop(event_loop);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WinitWindowId,
        event: WindowEvent,
    ) {
        let Some(handle) = self.handles_by_winit.get(&window_id).copied() else {
            return;
        };
        #[cfg(target_os = "windows")]
        let should_update_backend_surface =
            matches!(&event, WindowEvent::Resized(_) | WindowEvent::Occluded(_));

        let result = match event {
            WindowEvent::CloseRequested => self.ingest(PlatformFact::CloseRequested { handle }),
            WindowEvent::Destroyed => self.remove_native_window(handle),
            WindowEvent::Resized(size) => {
                let result = self.ingest_physical_size(handle, size);
                if size.width == 0 || size.height == 0 {
                    result.and_then(|()| {
                        self.ingest(PlatformFact::RenderabilityChanged {
                            handle,
                            renderability: Renderability::ZeroSize,
                        })
                    })
                } else {
                    result.and_then(|()| self.ingest(PlatformFact::Restored { handle }))
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.ingest(PlatformFact::ScaleFactorChanged {
                    handle,
                    scale_factor: scale_factor as f32,
                })
            }
            WindowEvent::RedrawRequested => {
                let result = self.ingest(PlatformFact::RedrawRequested { handle });
                #[cfg(target_os = "windows")]
                let result = result.and_then(|()| self.render_backend_frame(handle));
                result
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.ingest_pointer_move(window_id, handle, position)
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.ingest_mouse_button(handle, state, button)
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => self.ingest_keyboard_input(window_id, handle, event, is_synthetic),
            WindowEvent::ModifiersChanged(modifiers) => {
                self.ingest_modifiers_changed(window_id, handle, modifiers.state())
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                self.ingest_mouse_wheel(window_id, handle, delta, phase)
            }
            WindowEvent::Focused(focused) => self.ingest_window_focus(handle, focused),
            WindowEvent::Occluded(true) => self.ingest(PlatformFact::Minimized { handle }),
            WindowEvent::Occluded(false) => self.ingest(PlatformFact::Restored { handle }),
            _ => Ok(()),
        };

        if result.is_err() {
            self.exit_event_loop(event_loop);
            return;
        }
        #[cfg(target_os = "windows")]
        if should_update_backend_surface && self.update_backend_surface(handle).is_err() {
            self.exit_event_loop(event_loop);
            return;
        }
        if self.forward_platform_requests().is_err() {
            self.exit_event_loop(event_loop);
            return;
        }
        if self.should_exit() {
            self.exit_event_loop(event_loop);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let _ = self.ingest(PlatformFact::Wake);
        if self.forward_platform_requests().is_err() {
            self.exit_event_loop(event_loop);
            return;
        }
        if self.should_exit() {
            self.exit_event_loop(event_loop);
        }
    }
}

fn logical_cursor_position(position: winit::dpi::PhysicalPosition<f64>, scale: f64) -> LayoutPoint {
    let logical = position.to_logical::<f32>(scale);
    LayoutPoint::new(logical.x, logical.y)
}

fn logical_scroll_lines(x: f32, y: f32) -> ScrollDelta {
    ScrollDelta::lines(-x, -y)
}

fn logical_scroll_pixels(position: winit::dpi::PhysicalPosition<f64>, scale: f64) -> ScrollDelta {
    let logical = position.to_logical::<f32>(scale);
    ScrollDelta::pixels(-logical.x, -logical.y)
}

fn normalize_key(key: WinitKey) -> Key {
    match key {
        WinitKey::Character(value) => Key::character(value.to_string()),
        WinitKey::Named(value) => Key::named(format!("{value:?}")),
        WinitKey::Dead(value) => Key::Dead(value),
        WinitKey::Unidentified(_) => Key::Unidentified,
    }
}

fn normalize_physical_key(key: WinitPhysicalKey) -> PhysicalKey {
    match key {
        WinitPhysicalKey::Code(value) => PhysicalKey::code(format!("{value:?}")),
        WinitPhysicalKey::Unidentified(_) => PhysicalKey::Unidentified,
    }
}

fn normalize_modifiers(state: ModifiersState) -> Modifiers {
    Modifiers::new(
        state.shift_key(),
        state.control_key(),
        state.alt_key(),
        state.super_key(),
    )
}

fn normalize_scroll_phase(phase: TouchPhase) -> ScrollPhase {
    match phase {
        TouchPhase::Started => ScrollPhase::Started,
        TouchPhase::Moved => ScrollPhase::Moved,
        TouchPhase::Ended => ScrollPhase::Ended,
        TouchPhase::Cancelled => ScrollPhase::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use winit::dpi::PhysicalPosition;
    use winit::keyboard::ModifiersState;

    use super::{
        logical_cursor_position, logical_scroll_lines, logical_scroll_pixels, normalize_modifiers,
    };

    #[test]
    fn pointer_positions_are_converted_from_physical_to_logical() {
        let position = logical_cursor_position(PhysicalPosition::new(20.0, 10.0), 2.0);

        assert_eq!(position.x(), 10.0);
        assert_eq!(position.y(), 5.0);
    }

    #[test]
    fn wheel_line_deltas_are_converted_to_internal_offset_delta() {
        let delta = logical_scroll_lines(4.0, -2.5);

        assert_eq!(delta.x(), -4.0);
        assert_eq!(delta.y(), 2.5);
        assert_eq!(delta.unit_name(), "lines");
    }

    #[test]
    fn wheel_pixel_deltas_are_converted_to_internal_offset_delta() {
        let delta = logical_scroll_pixels(PhysicalPosition::new(9.0, 6.0), 3.0);

        assert_eq!(delta.x(), -3.0);
        assert_eq!(delta.y(), -2.0);
        assert_eq!(delta.unit_name(), "pixels");
    }

    #[test]
    fn modifiers_are_normalized_without_winit_leakage() {
        let modifiers = normalize_modifiers(ModifiersState::SHIFT.union(ModifiersState::ALT));

        assert!(modifiers.shift());
        assert!(!modifiers.ctrl());
        assert!(modifiers.alt());
        assert!(!modifiers.logo());
    }
}
