use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

use hashbrown::HashMap;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::WindowId as WinitWindowId;

use crate::app::{App, AppContext, LastWindowBehavior};
use crate::error::{Error, PlatformError};
use crate::input::{CaretRect, PointerButton, PointerEvent, PointerPhase, TextInputEvent, TextInputPurpose};
use crate::platform::wgpu::{RenderFramePackage, RenderOutcome, RenderSystem};
#[cfg(target_os = "linux")]
use crate::platform::window::native::linux::{
    backend_kind as linux_backend_kind, begin_client_decorations_resize,
    clear_client_decorations_cursor, route_for as linux_route_for,
    update_client_decorations_cursor,
};
#[cfg(target_os = "macos")]
use crate::platform::window::native::macos::{
    perform_window_frame_action as macos_perform_window_frame_action,
    titlebar_double_click as macos_titlebar_double_click,
    update_standard_window_buttons as macos_update_standard_window_buttons,
};
use crate::platform::window::{
    WindowCommand, active_displays, apply_geometry_patch, apply_post_create_state,
    current_display_id, current_frame_size, current_placement, current_position,
    update_hidden_titlebar_hit_test_state, window_attributes,
};
use crate::text_system::TextSystem;

use super::window_runtime::{
    FrameReasonMask, FrameSchedulerState, NativeWindowState, RuntimeWindow, WindowGenerationState,
    WindowMetrics, current_window_metrics, metrics_from_physical_size, metrics_from_scale_change,
    refresh_native_window_snapshot, update_native_window_metrics, update_native_window_occlusion,
    update_native_window_visibility, window_can_present, window_depends_on_dirty_views,
    window_metrics_changed,
};

#[derive(Debug, Clone, Copy)]
enum RunnerEvent {
    Wake,
}

type LaunchCallback = Box<dyn FnOnce(&mut AppContext<'_>) -> Result<(), Error>>;

struct AppRuntime {
    app: App,
    last_window_behavior: LastWindowBehavior,
    windows: HashMap<WinitWindowId, RuntimeWindow>,
    text_system: TextSystem,
    render_system: Option<RenderSystem>,
    pending_launch: Option<LaunchCallback>,
    startup_error: Option<Error>,
}

impl AppRuntime {
    fn new(
        app: App,
        last_window_behavior: LastWindowBehavior,
        pending_launch: impl FnOnce(&mut AppContext<'_>) -> Result<(), Error> + 'static,
    ) -> Self {
        Self {
            app,
            last_window_behavior,
            windows: HashMap::new(),
            text_system: TextSystem::new(),
            render_system: None,
            pending_launch: Some(Box::new(pending_launch)),
            startup_error: None,
        }
    }

    fn run_launch_if_needed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(on_launch) = self.pending_launch.take() else {
            return;
        };

        let runtime_ptr: *mut AppRuntime = self;
        let mut immediate_window_opener = move |request| unsafe {
            (&mut *runtime_ptr).create_window_request(event_loop, request)
        };
        let result = {
            let mut cx = AppContext::with_immediate_window_opener(
                &mut self.app,
                &mut immediate_window_opener,
            );
            on_launch(&mut cx)
        };

        if let Err(error) = result {
            self.startup_error = Some(error);
            event_loop.exit();
        }
    }

    fn create_window_request(
        &mut self,
        event_loop: &ActiveEventLoop,
        request: crate::app::PendingWindowRequest,
    ) -> Result<(), Error> {
        let displays = self.app.displays();
        let active_display = self.app.active_display().map(|display| display.id);
        let native_window = match event_loop.create_window(window_attributes(&request.options)) {
            Ok(window) => Arc::new(window),
            Err(error) => return Err(PlatformError::new(error.to_string()).into()),
        };
        apply_post_create_state(
            event_loop,
            native_window.as_ref(),
            &request.options,
            &displays,
            active_display,
        );

        let metrics = current_window_metrics(&native_window);
        let mut public_window = crate::window::WindowInfo::from_options(
            request.id,
            &request.options,
            crate::platform::window::WindowInfoSeed {
                content_size: metrics.logical_size,
                frame_size: current_frame_size(native_window.as_ref()),
                physical_size: metrics.physical_size,
                scale_factor: metrics.scale_factor,
                position: current_position(native_window.as_ref(), metrics.scale_factor),
                current_display: current_display_id(native_window.as_ref()),
            },
        );
        public_window.set_position(current_position(
            native_window.as_ref(),
            metrics.scale_factor,
        ));
        public_window.set_placement(current_placement(native_window.as_ref()));
        public_window.set_current_display(current_display_id(native_window.as_ref()));
        let template_root = (request.build_root)(&public_window, &mut self.app);
        let mut build_scratch = crate::element::SpecArena::new();
        let built = self
            .app
            .build_root_spec(&public_window, &template_root, &mut build_scratch)?;
        let mut retained_tree = crate::scene::RetainedTree::from_spec(&build_scratch, built.root);

        let render_state = if let Some(render_system) = self.render_system.as_mut() {
            let surface = render_system.create_surface_for_window(native_window.clone())?;
            render_system.create_window_state(
                native_window.clone(),
                surface,
                metrics.physical_size,
            )?
        } else {
            let (render_system, render_state) =
                RenderSystem::new(native_window.clone(), metrics.physical_size)?;
            self.render_system = Some(render_system);
            render_state
        };
        retained_tree.compute_layout(metrics.logical_size, &mut self.text_system);
        update_hidden_titlebar_hit_test_state(
            native_window.as_ref(),
            public_window.scale_factor(),
            &retained_tree.collect_window_frame_areas(),
        );
        let compiled_scene = retained_tree.compile_scene();
        let initial_visibility = public_window.visible();

        self.windows.insert(
            native_window.id(),
            RuntimeWindow {
                _internal_id: request.id,
                public_window,
                native_window: native_window.clone(),
                template_root,
                referenced_views: built.referenced_views,
                build_scratch,
                retained_tree,
                compiled_scene,
                render_state,
                native_state: NativeWindowState::new(
                    metrics,
                    initial_visibility,
                    native_window.is_minimized().unwrap_or(false),
                    false,
                ),
                frame_scheduler: FrameSchedulerState::new(),
                generation_state: WindowGenerationState::new_synced(),
                cursor_position: None,
                focus_manager: super::input::FocusManager::new(),
                input_router: super::input::InputRouter::default(),
                text_input_target: super::input::TextInputTarget::default(),
                #[cfg(target_os = "linux")]
                linux_route: linux_route_for(linux_backend_kind(event_loop), &request.options),
                #[cfg(target_os = "macos")]
                last_drag_click: None,
            },
        );
        if let Some(runtime_window) = self.windows.get_mut(&native_window.id()) {
            mark_window_pending(runtime_window, FrameReasonMask::SURFACE);
        }
        Ok(())
    }

    fn process_window_requests(&mut self, event_loop: &ActiveEventLoop) {
        for request in self.app.drain_window_requests() {
            if let Err(error) = self.create_window_request(event_loop, request) {
                log::error!("failed to create window: {error}");
            }
        }
    }

    fn process_runtime_updates(&mut self) {
        let runtime = match self.app.process_runtime() {
            Ok(runtime) => runtime,
            Err(error) => {
                log::error!("runtime processing failed: {error}");
                return;
            }
        };

        if runtime.dirty_views.is_empty() {
            return;
        }

        let (windows, text_system) = (&mut self.windows, &mut self.text_system);
        for runtime_window in windows.values_mut() {
            if !window_depends_on_dirty_views(runtime_window, &runtime.dirty_views) {
                continue;
            }

            let built = match self.app.build_root_spec(
                &runtime_window.public_window,
                &runtime_window.template_root,
                &mut runtime_window.build_scratch,
            ) {
                Ok(value) => value,
                Err(error) => {
                    log::error!("failed to rebuild root element: {error}");
                    continue;
                }
            };
            runtime_window.referenced_views = built.referenced_views;
            let dirty = runtime_window
                .retained_tree
                .update_from_spec(&runtime_window.build_scratch, built.root);
            sync_window_frame_hit_test(runtime_window);
            sync_focus_after_tree_update(runtime_window);

            if dirty.needs_layout() {
                runtime_window
                    .retained_tree
                    .compute_layout(runtime_window.public_window.content_size(), text_system);
                sync_window_frame_hit_test(runtime_window);
                sync_focus_after_tree_update(runtime_window);
            }
            if dirty.needs_scene_compile() {
                runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
                runtime_window.generation_state.mark_scene_compiled();
                mark_window_pending(runtime_window, FrameReasonMask::RUNTIME_DIRTY);
            }
        }
    }

    fn process_window_commands(&mut self, event_loop: &ActiveEventLoop) {
        let displays = self.app.displays();
        let active_display = self.app.active_display().map(|display| display.id);
        let commands = self.app.drain_window_commands();
        for (window_id, command) in commands {
            let native_id = self
                .windows
                .iter()
                .find(|(_, runtime_window)| runtime_window._internal_id == window_id)
                .map(|(native_id, _)| *native_id);

            let Some(native_id) = native_id else {
                continue;
            };

            if matches!(command, WindowCommand::Close) {
                self.close_window(event_loop, native_id);
                continue;
            }

            let Some(runtime_window) = self.windows.get_mut(&native_id) else {
                continue;
            };

            match command {
                WindowCommand::Close => unreachable!(),
                WindowCommand::Focus => runtime_window.native_window.focus_window(),
                WindowCommand::RequestRedraw => {
                    mark_window_pending(runtime_window, FrameReasonMask::USER);
                }
                WindowCommand::SetTitle(title) => {
                    runtime_window.native_window.set_title(title.as_ref());
                    runtime_window.public_window.set_title(title);
                }
                WindowCommand::SetGeometry(patch) => {
                    apply_geometry_patch(
                        runtime_window.native_window.as_ref(),
                        &patch,
                        &displays,
                        active_display,
                    );
                    refresh_window_native_state(runtime_window, self.render_system.as_mut());
                }
                WindowCommand::SetVisible(visible) => {
                    runtime_window.native_window.set_visible(visible);
                    runtime_window.public_window.set_visible(visible);
                    let visibility_update =
                        update_native_window_visibility(runtime_window, visible);
                    apply_native_window_update(runtime_window, visibility_update);
                }
                WindowCommand::SetResizable(resizable) => {
                    runtime_window.native_window.set_resizable(resizable);
                    runtime_window.public_window.set_resizable(resizable);
                }
                WindowCommand::Maximize => runtime_window.native_window.set_maximized(true),
                WindowCommand::Unmaximize => runtime_window.native_window.set_maximized(false),
                WindowCommand::Fullscreen(display) => {
                    let placement = crate::window::WindowPlacement::Fullscreen;
                    let selector = display.unwrap_or(crate::window::DisplaySelector::Active);
                    apply_geometry_patch(
                        runtime_window.native_window.as_ref(),
                        &crate::window::WindowGeometryPatch {
                            position: None,
                            size: None,
                            min_size: None,
                            max_size: None,
                            placement: Some(placement),
                        },
                        &displays,
                        match selector {
                            crate::window::DisplaySelector::Primary => displays
                                .iter()
                                .find(|display| display.is_primary)
                                .map(|d| d.id),
                            crate::window::DisplaySelector::Active => active_display,
                            crate::window::DisplaySelector::ById(id) => Some(id),
                        },
                    );
                }
                WindowCommand::ExitFullscreen => runtime_window.native_window.set_fullscreen(None),
                WindowCommand::Minimize => runtime_window.native_window.set_minimized(true),
            }
        }
    }

    fn drive_pending_presentations(&mut self) {
        let (windows, render_system) = (&mut self.windows, &mut self.render_system);
        let window_ids = windows.keys().copied().collect::<Vec<_>>();
        for window_id in window_ids {
            let Some(runtime_window) = windows.get_mut(&window_id) else {
                continue;
            };
            refresh_window_native_state(runtime_window, render_system.as_mut());
            if runtime_window.frame_scheduler.has_pending() && window_can_present(runtime_window) {
                request_window_redraw(runtime_window);
            }
        }
    }

    fn close_window(&mut self, event_loop: &ActiveEventLoop, window_id: WinitWindowId) {
        self.windows.remove(&window_id);
        if self.windows.is_empty()
            && matches!(self.last_window_behavior, LastWindowBehavior::ExitEventLoop)
        {
            event_loop.exit();
        }
    }
}

impl ApplicationHandler<RunnerEvent> for AppRuntime {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.app.set_displays(active_displays(event_loop));
        self.run_launch_if_needed(event_loop);
        self.process_window_requests(event_loop);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: RunnerEvent) {
        match event {
            RunnerEvent::Wake => {
                self.app.set_displays(active_displays(event_loop));
                self.run_launch_if_needed(event_loop);
                self.process_window_requests(event_loop);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WinitWindowId,
        event: WindowEvent,
    ) {
        // TODO: 部分事件的上行
        match event {
            WindowEvent::CloseRequested => self.close_window(event_loop, window_id), // TODO: 转发到on_close_requested，允许上层处理
            WindowEvent::Focused(focused) => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    runtime_window.public_window.set_focused(focused);
                    if focused {
                        let current_display =
                            current_display_id(runtime_window.native_window.as_ref());
                        runtime_window
                            .public_window
                            .set_current_display(current_display);
                        self.app.set_active_display(current_display);
                    }
                }
            }
            WindowEvent::Moved(_) => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    runtime_window.public_window.set_position(current_position(
                        runtime_window.native_window.as_ref(),
                        runtime_window.public_window.scale_factor(),
                    ));
                    runtime_window
                        .public_window
                        .set_placement(current_placement(runtime_window.native_window.as_ref()));
                    let current_display = current_display_id(runtime_window.native_window.as_ref());
                    runtime_window
                        .public_window
                        .set_current_display(current_display);
                    self.app.set_active_display(current_display);
                }
            }
            WindowEvent::Resized(size) => {
                let (windows, render_system) = (&mut self.windows, &mut self.render_system);
                if let Some(runtime_window) = windows.get_mut(&window_id) {
                    let metrics = metrics_from_physical_size(runtime_window, size);
                    sync_window_metrics(runtime_window, metrics, render_system.as_mut());
                }
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                mut inner_size_writer,
            } => {
                if let Some(runtime_window) = self.windows.get(&window_id) {
                    let requested = runtime_window.native_window.inner_size();
                    let _ = inner_size_writer.request_inner_size(requested);
                }
                let (windows, render_system) = (&mut self.windows, &mut self.render_system);
                if let Some(runtime_window) = windows.get_mut(&window_id) {
                    let metrics = metrics_from_scale_change(runtime_window, scale_factor);
                    sync_window_metrics(runtime_window, metrics, render_system.as_mut());
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    runtime_window.cursor_position = Some(position);
                    let logical = logical_point_from_cursor(runtime_window, position);
                    let _ = runtime_window.input_router.pointer_event(
                        &runtime_window.retained_tree,
                        PointerEvent {
                            phase: PointerPhase::Move,
                            position: logical,
                            button: None,
                            delta: None,
                        },
                    );
                    #[cfg(target_os = "linux")]
                    {
                        update_client_decorations_cursor(
                            runtime_window.linux_route,
                            runtime_window.native_window.as_ref(),
                            &runtime_window.public_window,
                            position,
                        );
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    runtime_window.cursor_position = None;
                    runtime_window.input_router.last_hit = None;
                    #[cfg(target_os = "linux")]
                    {
                        clear_client_decorations_cursor(
                            runtime_window.linux_route,
                            runtime_window.native_window.as_ref(),
                        );
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id)
                    && let Some(cursor_position) = runtime_window.cursor_position
                {
                    let logical = logical_point_from_cursor(runtime_window, cursor_position);
                    let hit = runtime_window.input_router.pointer_event(
                        &runtime_window.retained_tree,
                        PointerEvent {
                            phase: if state == ElementState::Pressed {
                                PointerPhase::Down
                            } else {
                                PointerPhase::Up
                            },
                            position: logical,
                            button: Some(pointer_button(button)),
                            delta: None,
                        },
                    );

                    if state == ElementState::Pressed && button == MouseButton::Left {
                        let focused = hit.focusable;
                        if runtime_window.focus_manager.set_focused(focused) {
                            sync_text_input_target(runtime_window);
                            apply_ime_target(runtime_window);
                            mark_window_pending(runtime_window, FrameReasonMask::USER);
                        }
                    }
                }

                if state != ElementState::Pressed || button != MouseButton::Left {
                    return;
                }

                #[cfg(target_os = "linux")]
                let mut handled_window_area = false;
                if let Some(runtime_window) = self.windows.get_mut(&window_id)
                    && let Some(area) = window_frame_area_at_cursor(runtime_window)
                {
                    #[cfg(target_os = "linux")]
                    {
                        handled_window_area = true;
                    }
                    match area {
                        crate::element::WindowFrameArea::Drag => {
                            #[cfg(target_os = "macos")]
                            {
                                let handled_native = handle_macos_drag_double_click(runtime_window);
                                if !handled_native {
                                    let _ = runtime_window.native_window.drag_window();
                                }
                            }
                            #[cfg(not(target_os = "macos"))]
                            let _ = runtime_window.native_window.drag_window();
                        }
                        crate::element::WindowFrameArea::Close => {
                            #[cfg(target_os = "macos")]
                            {
                                let handled_native = macos_perform_window_frame_action(
                                    runtime_window.native_window.as_ref(),
                                    area,
                                );
                                if !handled_native {
                                    self.close_window(event_loop, window_id);
                                }
                            }
                            #[cfg(not(target_os = "macos"))]
                            self.close_window(event_loop, window_id);
                        }
                        crate::element::WindowFrameArea::Maximize => {
                            #[cfg(target_os = "macos")]
                            {
                                let handled_native = macos_perform_window_frame_action(
                                    runtime_window.native_window.as_ref(),
                                    area,
                                );
                                if !handled_native {
                                    runtime_window.native_window.set_maximized(
                                        !runtime_window.native_window.is_maximized(),
                                    );
                                }
                            }
                            #[cfg(not(target_os = "macos"))]
                            runtime_window
                                .native_window
                                .set_maximized(!runtime_window.native_window.is_maximized());
                        }
                        crate::element::WindowFrameArea::Minimize => {
                            #[cfg(target_os = "macos")]
                            {
                                let handled_native = macos_perform_window_frame_action(
                                    runtime_window.native_window.as_ref(),
                                    area,
                                );
                                if !handled_native {
                                    runtime_window.native_window.set_minimized(true);
                                }
                            }
                            #[cfg(not(target_os = "macos"))]
                            runtime_window.native_window.set_minimized(true);
                        }
                    }
                }
                #[cfg(target_os = "linux")]
                {
                    if !handled_window_area
                        && let Some(runtime_window) = self.windows.get_mut(&window_id)
                    {
                        begin_client_decorations_resize(
                            runtime_window.linux_route,
                            runtime_window.native_window.as_ref(),
                            &runtime_window.public_window,
                            runtime_window.cursor_position,
                        );
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id)
                    && let Some(cursor_position) = runtime_window.cursor_position
                {
                    let logical = logical_point_from_cursor(runtime_window, cursor_position);
                    let delta = match delta {
                        MouseScrollDelta::LineDelta(x, y) => crate::style::Point::new(
                            crate::style::px(x * 12.0),
                            crate::style::px(y * 12.0),
                        ),
                        MouseScrollDelta::PixelDelta(position) => crate::style::Point::new(
                            crate::style::px(position.x as f32),
                            crate::style::px(position.y as f32),
                        ),
                    };
                    let _ = runtime_window.input_router.pointer_event(
                        &runtime_window.retained_tree,
                        PointerEvent {
                            phase: PointerPhase::Wheel,
                            position: logical,
                            button: None,
                            delta: Some(delta),
                        },
                    );
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id)
                    && event.state == ElementState::Pressed
                {
                    match &event.logical_key {
                        WinitKey::Named(NamedKey::Tab) => {
                            let next = runtime_window
                                .retained_tree
                                .next_focusable_node(runtime_window.focus_manager.focused());
                            if runtime_window.focus_manager.set_focused(next) {
                                sync_text_input_target(runtime_window);
                                apply_ime_target(runtime_window);
                                mark_window_pending(runtime_window, FrameReasonMask::USER);
                            }
                        }
                        WinitKey::Named(NamedKey::Escape) => {
                            if runtime_window.focus_manager.set_focused(None) {
                                sync_text_input_target(runtime_window);
                                apply_ime_target(runtime_window);
                                mark_window_pending(runtime_window, FrameReasonMask::USER);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Ime(event) => {
                let mut dispatched = None;
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    dispatched = match event {
                        Ime::Enabled => None,
                        Ime::Preedit(text, _) => {
                            let preedit = if text.is_empty() { None } else { Some(text) };
                            runtime_window.text_input_target.preedit = preedit.clone();
                            match (
                                text_input_owner_view_id(runtime_window),
                                runtime_window.text_input_target.input_id,
                            ) {
                                (Some(owner_view_id), Some(target)) => Some((
                                    owner_view_id,
                                    TextInputEvent::Preedit {
                                        source: owner_view_id,
                                        target,
                                        text: preedit.map(Into::into),
                                    },
                                )),
                                _ => None,
                            }
                        }
                        Ime::Commit(text) => {
                            runtime_window.text_input_target.preedit = None;
                            if text.is_empty() {
                                None
                            } else {
                                match (
                                    text_input_owner_view_id(runtime_window),
                                    runtime_window.text_input_target.input_id,
                                ) {
                                    (Some(owner_view_id), Some(target)) => Some((
                                        owner_view_id,
                                        TextInputEvent::Commit {
                                            source: owner_view_id,
                                            target,
                                            text: text.into(),
                                        },
                                    )),
                                    _ => None,
                                }
                            }
                        }
                        Ime::Disabled => {
                            runtime_window.text_input_target.preedit = None;
                            match (
                                text_input_owner_view_id(runtime_window),
                                runtime_window.text_input_target.input_id,
                            ) {
                                (Some(owner_view_id), Some(target)) => Some((
                                    owner_view_id,
                                    TextInputEvent::Preedit {
                                        source: owner_view_id,
                                        target,
                                        text: None,
                                    },
                                )),
                                _ => None,
                            }
                        }
                    };
                }
                if let Some((owner_view_id, text_input_event)) = dispatched {
                    let _ = self
                        .app
                        .dispatch_text_input_event(owner_view_id, &text_input_event);
                }
            }
            WindowEvent::Occluded(occluded) => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    let occlusion_update = update_native_window_occlusion(runtime_window, occluded);
                    if let Some(render_system) = self.render_system.as_ref() {
                        render_system
                            .note_surface_occlusion(&mut runtime_window.render_state, occluded);
                    }
                    apply_native_window_update(runtime_window, occlusion_update);
                }
            }
            WindowEvent::RedrawRequested => {
                let (windows, text_system, render_system) = (
                    &mut self.windows,
                    &mut self.text_system,
                    &mut self.render_system,
                );
                if let (Some(render_system), Some(runtime_window)) =
                    (render_system.as_mut(), windows.get_mut(&window_id))
                {
                    runtime_window.frame_scheduler.begin_redraw();
                    refresh_window_native_state(runtime_window, Some(render_system));
                    sync_window_scene_to_latest_metrics(runtime_window, text_system);
                    let frame_package = RenderFramePackage {
                        scene: &runtime_window.compiled_scene,
                        metrics_generation: runtime_window.generation_state.metrics_generation,
                        scene_generation: runtime_window.generation_state.scene_generation,
                        target_surface_generation: runtime_window.render_state.surface_generation(),
                        scale_factor: runtime_window.public_window.scale_factor(),
                    };
                    match render_system.render(
                        &mut runtime_window.render_state,
                        frame_package,
                        text_system,
                    ) {
                        Ok(RenderOutcome::Presented) => {
                            runtime_window.generation_state.mark_presented();
                            runtime_window.frame_scheduler.mark_presented();
                        }
                        Ok(RenderOutcome::PresentedSuboptimal) => {
                            runtime_window.generation_state.mark_presented();
                            mark_window_pending(
                                runtime_window,
                                FrameReasonMask::SURFACE | FrameReasonMask::METRICS,
                            );
                        }
                        Ok(RenderOutcome::Reconfigure) => {
                            mark_window_pending(runtime_window, FrameReasonMask::SURFACE);
                        }
                        Ok(RenderOutcome::RecreateSurface) => {
                            let _ =
                                render_system.recreate_surface(&mut runtime_window.render_state);
                            mark_window_pending(runtime_window, FrameReasonMask::SURFACE);
                        }
                        Ok(RenderOutcome::Unavailable) => {
                            runtime_window
                                .frame_scheduler
                                .mark_pending(FrameReasonMask::SURFACE);
                        }
                        Err(error) => {
                            log::error!("render failed: {error}");
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.app.set_displays(active_displays(event_loop));
        self.process_window_commands(event_loop);
        self.process_window_requests(event_loop);
        self.process_runtime_updates();
        self.drive_pending_presentations();
    }
}

fn mark_window_pending(runtime_window: &mut RuntimeWindow, reasons: FrameReasonMask) {
    runtime_window.frame_scheduler.mark_pending(reasons);
    request_window_redraw(runtime_window);
}

fn window_frame_area_at_cursor(
    runtime_window: &RuntimeWindow,
) -> Option<crate::element::WindowFrameArea> {
    if runtime_window.public_window.show_titlebar() {
        return None;
    }

    let cursor_position = runtime_window.cursor_position?;

    let scale = runtime_window.public_window.scale_factor() as f32;
    let logical_point = crate::style::Point::new(
        crate::style::px(cursor_position.x as f32 / scale),
        crate::style::px(cursor_position.y as f32 / scale),
    );

    runtime_window
        .retained_tree
        .window_frame_area_at(logical_point)
}

fn sync_window_metrics(
    runtime_window: &mut RuntimeWindow,
    metrics: WindowMetrics,
    render_system: Option<&mut RenderSystem>,
) {
    let native_update = update_native_window_metrics(runtime_window, metrics);
    let metrics_changed = native_update.metrics_changed;
    let physical_size_changed = native_update.physical_size_changed;
    runtime_window.public_window.set_content_metrics(
        metrics.logical_size,
        current_frame_size(runtime_window.native_window.as_ref()),
        metrics.physical_size,
        metrics.scale_factor,
    );
    runtime_window.public_window.set_position(current_position(
        runtime_window.native_window.as_ref(),
        metrics.scale_factor,
    ));
    runtime_window
        .public_window
        .set_placement(current_placement(runtime_window.native_window.as_ref()));
    runtime_window
        .public_window
        .set_current_display(current_display_id(runtime_window.native_window.as_ref()));

    if physical_size_changed && let Some(render_system) = render_system {
        render_system.note_surface_resize(&mut runtime_window.render_state, metrics.physical_size);
    }
    if native_update.became_unavailable {
        runtime_window.frame_scheduler.mark_unavailable();
    }
    if metrics_changed {
        runtime_window.generation_state.note_metrics_change();
        mark_window_pending(
            runtime_window,
            FrameReasonMask::METRICS
                | if physical_size_changed {
                    FrameReasonMask::SURFACE
                } else {
                    FrameReasonMask::empty()
                },
        );
    }
    if native_update.became_presentable && !metrics_changed {
        mark_window_pending(
            runtime_window,
            FrameReasonMask::VISIBILITY | FrameReasonMask::SURFACE,
        );
    }
}

fn refresh_window_native_state(
    runtime_window: &mut RuntimeWindow,
    render_system: Option<&mut RenderSystem>,
) {
    let metrics = current_window_metrics(&runtime_window.native_window);
    runtime_window.public_window.set_position(current_position(
        runtime_window.native_window.as_ref(),
        metrics.scale_factor,
    ));
    runtime_window
        .public_window
        .set_placement(current_placement(runtime_window.native_window.as_ref()));
    runtime_window
        .public_window
        .set_current_display(current_display_id(runtime_window.native_window.as_ref()));

    let native_update = refresh_native_window_snapshot(runtime_window, metrics);
    if window_metrics_changed(runtime_window, metrics) {
        sync_window_metrics(runtime_window, metrics, render_system);
        return;
    }

    if native_update.became_unavailable {
        runtime_window.frame_scheduler.mark_unavailable();
    }
    if native_update.became_presentable {
        mark_window_pending(
            runtime_window,
            FrameReasonMask::VISIBILITY | FrameReasonMask::SURFACE,
        );
    }
}

fn apply_native_window_update(
    runtime_window: &mut RuntimeWindow,
    update: super::window_runtime::NativeWindowUpdate,
) {
    if update.became_unavailable {
        runtime_window.frame_scheduler.mark_unavailable();
    }
    if update.became_presentable {
        mark_window_pending(
            runtime_window,
            FrameReasonMask::VISIBILITY | FrameReasonMask::SURFACE,
        );
    }
}

fn request_window_redraw(runtime_window: &mut RuntimeWindow) {
    if !window_can_present(runtime_window) {
        runtime_window.frame_scheduler.mark_unavailable();
        return;
    }

    if runtime_window.frame_scheduler.redraw_requested {
        return;
    }

    runtime_window.native_window.request_redraw();
    runtime_window.frame_scheduler.mark_redraw_requested();
}

fn sync_window_scene_to_latest_metrics(
    runtime_window: &mut RuntimeWindow,
    text_system: &mut TextSystem,
) {
    if !runtime_window.generation_state.scene_is_stale() {
        return;
    }

    runtime_window
        .retained_tree
        .compute_layout(runtime_window.public_window.content_size(), text_system);
    sync_window_frame_hit_test(runtime_window);
    sync_focus_after_tree_update(runtime_window);
    runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
    runtime_window.generation_state.mark_scene_compiled();
}

fn sync_window_frame_hit_test(runtime_window: &RuntimeWindow) {
    let areas = runtime_window.retained_tree.collect_window_frame_areas();
    update_hidden_titlebar_hit_test_state(
        runtime_window.native_window.as_ref(),
        runtime_window.public_window.scale_factor(),
        &areas,
    );
    #[cfg(target_os = "macos")]
    {
        macos_update_standard_window_buttons(
            runtime_window.native_window.as_ref(),
            runtime_window.public_window.scale_factor(),
            &areas,
        );
    }
}

fn sync_text_input_target(runtime_window: &mut RuntimeWindow) {
    let focused = runtime_window.focus_manager.focused();
    let text_input_node = focused.filter(|node| {
        runtime_window
            .retained_tree
            .retained_node(*node)
            .interaction
            .text_input
            .is_some()
            && !runtime_window
                .retained_tree
                .retained_node(*node)
                .semantics
                .disabled
    });

    runtime_window.text_input_target.node = text_input_node;
    runtime_window.text_input_target.input_id = text_input_node.and_then(|node| {
        runtime_window
            .retained_tree
            .retained_node(node)
            .input_id
    });
    runtime_window.text_input_target.purpose = text_input_node
        .and_then(|node| {
            runtime_window
                .retained_tree
                .retained_node(node)
                .interaction
                .text_input
                .as_ref()
                .map(|state| state.purpose)
        })
        .unwrap_or(TextInputPurpose::Normal);
    runtime_window.text_input_target.caret_rect = text_input_node.map(|node| {
        let layout = runtime_window.retained_tree.absolute_layout_box(node);
        CaretRect {
            origin: crate::style::point(crate::style::px(layout.x), crate::style::px(layout.y)),
            size: crate::style::size(
                crate::style::px(1.0),
                crate::style::px(layout.height.max(1.0)),
            ),
        }
    });
}

fn text_input_owner_view_id(runtime_window: &RuntimeWindow) -> Option<u64> {
    let focused = runtime_window.focus_manager.focused()?;
    runtime_window
        .retained_tree
        .retained_node(focused)
        .owner_view_id
}

fn sync_focus_after_tree_update(runtime_window: &mut RuntimeWindow) {
    let focused = runtime_window.focus_manager.focused();
    let next_focused = focused.filter(|node| {
        runtime_window
            .retained_tree
            .try_retained_node(*node)
            .is_some_and(|retained| {
                retained.interaction.focus_policy != crate::input::FocusPolicy::None
                    && !retained.semantics.disabled
                    && !matches!(retained.style.layout.display, crate::style::Display::None)
                    && retained.layout.width > 0.0
                    && retained.layout.height > 0.0
            })
    });
    runtime_window.focus_manager.set_focused(next_focused);
    sync_text_input_target(runtime_window);
    apply_ime_target(runtime_window);
}

fn apply_ime_target(runtime_window: &mut RuntimeWindow) {
    let allowed = runtime_window.text_input_target.node.is_some();
    runtime_window.native_window.set_ime_allowed(allowed);
    if allowed && let Some(caret) = runtime_window.text_input_target.caret_rect {
        let scale = runtime_window.public_window.scale_factor() as f32;
        runtime_window.native_window.set_ime_cursor_area(
            winit::dpi::PhysicalPosition::new(
                (caret.origin.x.get() * scale).round() as i32,
                (caret.origin.y.get() * scale).round() as i32,
            ),
            PhysicalSize::new(
                (caret.size.width.get() * scale).round().max(1.0) as u32,
                (caret.size.height.get() * scale).round().max(1.0) as u32,
            ),
        );
    }
}

fn logical_point_from_cursor(
    runtime_window: &RuntimeWindow,
    position: winit::dpi::PhysicalPosition<f64>,
) -> crate::style::Point<crate::style::Px> {
    let scale = runtime_window.public_window.scale_factor() as f32;
    crate::style::Point::new(
        crate::style::px(position.x as f32 / scale),
        crate::style::px(position.y as f32 / scale),
    )
}

fn pointer_button(button: MouseButton) -> PointerButton {
    match button {
        MouseButton::Left => PointerButton::Primary,
        MouseButton::Right => PointerButton::Secondary,
        MouseButton::Middle => PointerButton::Middle,
        MouseButton::Back => PointerButton::Back,
        MouseButton::Forward => PointerButton::Forward,
        MouseButton::Other(value) => PointerButton::Other(value),
    }
}

#[cfg(target_os = "macos")]
fn handle_macos_drag_double_click(runtime_window: &mut RuntimeWindow) -> bool {
    let Some(cursor_position) = runtime_window.cursor_position else {
        return false;
    };
    let now = Instant::now();
    let is_double_click =
        runtime_window
            .last_drag_click
            .is_some_and(|(last_time, last_position)| {
                now.duration_since(last_time) <= Duration::from_millis(500)
                    && (last_position.x - cursor_position.x).abs() <= 4.0
                    && (last_position.y - cursor_position.y).abs() <= 4.0
            });
    runtime_window.last_drag_click = Some((now, cursor_position));

    if is_double_click {
        runtime_window.last_drag_click = None;
        return macos_titlebar_double_click(runtime_window.native_window.as_ref());
    }

    false
}

pub(crate) fn run_application(
    last_window_behavior: LastWindowBehavior,
    on_launch: impl FnOnce(&mut AppContext<'_>) -> Result<(), Error> + 'static,
) -> Result<(), Error> {
    let event_loop = EventLoop::<RunnerEvent>::with_user_event()
        .build()
        .map_err(|error| PlatformError::new(error.to_string()))?;
    let proxy: EventLoopProxy<RunnerEvent> = event_loop.create_proxy();

    let mut app = App::new(Vec::new());
    app.set_wake_handle(Some(Arc::new(move || {
        let _ = proxy.send_event(RunnerEvent::Wake);
    })));

    let mut runtime = AppRuntime::new(app, last_window_behavior, on_launch);
    event_loop
        .run_app(&mut runtime)
        .map_err(|error| PlatformError::new(error.to_string()))?;

    if let Some(error) = runtime.startup_error {
        return Err(error);
    }

    Ok(())
}
