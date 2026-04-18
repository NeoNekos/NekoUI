use std::sync::Arc;

use hashbrown::HashMap;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{WindowAttributes, WindowId as WinitWindowId};

use crate::app::{App, LastWindowBehavior};
use crate::error::{Error, PlatformError};
use crate::platform::wgpu::{RenderOutcome, RenderSystem};
use crate::text_system::TextSystem;

use super::window_runtime::{
    RuntimeWindow, WindowMetrics, current_window_metrics, metrics_from_physical_size,
    metrics_from_scale_change, window_can_present, window_depends_on_dirty_views,
    window_metrics_changed,
};

#[derive(Debug, Clone, Copy)]
enum RunnerEvent {
    Wake,
}

struct AppRuntime {
    app: App,
    last_window_behavior: LastWindowBehavior,
    windows: HashMap<WinitWindowId, RuntimeWindow>,
    text_system: TextSystem,
    render_system: Option<RenderSystem>,
}

impl AppRuntime {
    fn new(app: App, last_window_behavior: LastWindowBehavior) -> Self {
        Self {
            app,
            last_window_behavior,
            windows: HashMap::new(),
            text_system: TextSystem::new(),
            render_system: None,
        }
    }

    fn process_window_requests(&mut self, event_loop: &ActiveEventLoop) {
        for request in self.app.drain_window_requests() {
            let size = request.options.size_value();
            let attributes = WindowAttributes::default()
                .with_title(request.options.title_str().to_string())
                .with_inner_size(LogicalSize::new(
                    f64::from(size.width),
                    f64::from(size.height),
                ));

            let native_window = match event_loop.create_window(attributes) {
                Ok(window) => Arc::new(window),
                Err(error) => {
                    log::error!("failed to create window: {error}");
                    continue;
                }
            };

            let metrics = current_window_metrics(&native_window);
            let mut public_window = App::make_runtime_window(
                request.id,
                &request.options,
                metrics.logical_size,
                metrics.physical_size,
                metrics.scale_factor,
            );
            let template_root = (request.build_root)(&mut public_window, &mut self.app);
            let mut build_scratch = crate::element::SpecArena::new();
            let built = match self.app.build_root_spec(
                &mut public_window,
                &template_root,
                &mut build_scratch,
            ) {
                Ok(value) => value,
                Err(error) => {
                    log::error!("failed to resolve root element: {error}");
                    continue;
                }
            };
            let mut retained_tree =
                crate::scene::RetainedTree::from_spec(&build_scratch, built.root);

            let render_state = if let Some(render_system) = self.render_system.as_mut() {
                let surface = match render_system.create_surface_for_window(native_window.clone()) {
                    Ok(surface) => surface,
                    Err(error) => {
                        log::error!("failed to create surface: {error}");
                        continue;
                    }
                };
                match render_system.create_window_state(surface, metrics.physical_size) {
                    Ok(render_state) => render_state,
                    Err(error) => {
                        log::error!("failed to create window render state: {error}");
                        continue;
                    }
                }
            } else {
                let (render_system, render_state) =
                    match RenderSystem::new(native_window.clone(), metrics.physical_size) {
                        Ok(value) => value,
                        Err(error) => {
                            log::error!("failed to initialize render system: {error}");
                            continue;
                        }
                    };
                self.render_system = Some(render_system);
                render_state
            };
            retained_tree.compute_layout(metrics.logical_size, &mut self.text_system);
            let compiled_scene = retained_tree.compile_scene();

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
                    presentation_pending: true,
                    metrics_scene_sync_pending: false,
                    occluded: false,
                },
            );
            native_window.request_redraw();
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
                &mut runtime_window.public_window,
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

            if dirty.needs_layout() {
                runtime_window
                    .retained_tree
                    .compute_layout(runtime_window.public_window.size(), text_system);
            }
            if dirty.needs_scene_compile() {
                runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
                mark_window_pending(runtime_window, "dirty_scene_compile");
            }
        }
    }

    fn drive_pending_presentations(&mut self) {
        let (windows, text_system, render_system) = (
            &mut self.windows,
            &mut self.text_system,
            &mut self.render_system,
        );
        let window_ids = windows.keys().copied().collect::<Vec<_>>();
        for window_id in window_ids {
            let Some(runtime_window) = windows.get_mut(&window_id) else {
                continue;
            };
            sync_window_to_native_state(runtime_window, text_system, render_system.as_mut());
            if runtime_window.presentation_pending && window_can_present(runtime_window) {
                runtime_window.native_window.request_redraw();
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
        self.process_window_requests(event_loop);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: RunnerEvent) {
        match event {
            RunnerEvent::Wake => self.process_window_requests(event_loop),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WinitWindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => self.close_window(event_loop, window_id),
            WindowEvent::Resized(size) => {
                let (windows, text_system, render_system) = (
                    &mut self.windows,
                    &mut self.text_system,
                    &mut self.render_system,
                );
                if let Some(runtime_window) = windows.get_mut(&window_id) {
                    let metrics = metrics_from_physical_size(runtime_window, size);
                    sync_window_metrics(
                        runtime_window,
                        metrics,
                        text_system,
                        render_system.as_mut(),
                        false,
                    );
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
                let (windows, text_system, render_system) = (
                    &mut self.windows,
                    &mut self.text_system,
                    &mut self.render_system,
                );
                if let Some(runtime_window) = windows.get_mut(&window_id) {
                    let metrics = metrics_from_scale_change(runtime_window, scale_factor);
                    sync_window_metrics(
                        runtime_window,
                        metrics,
                        text_system,
                        render_system.as_mut(),
                        false,
                    );
                }
            }
            WindowEvent::Occluded(occluded) => {
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    runtime_window.occluded = occluded;
                    if !occluded {
                        mark_window_pending(runtime_window, "occluded_false");
                    }
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
                    sync_window_to_native_state(runtime_window, text_system, Some(render_system));
                    let metrics_scene_sync_pending = runtime_window.metrics_scene_sync_pending;
                    match render_system.render(
                        &mut runtime_window.render_state,
                        &runtime_window.compiled_scene,
                        text_system,
                        &runtime_window.native_window,
                        runtime_window.public_window.scale_factor(),
                    ) {
                        Ok(RenderOutcome::Presented) => {
                            if metrics_scene_sync_pending {
                                runtime_window.retained_tree.compute_layout(
                                    runtime_window.public_window.size(),
                                    text_system,
                                );
                                runtime_window.compiled_scene =
                                    runtime_window.retained_tree.compile_scene();
                                runtime_window.metrics_scene_sync_pending = false;
                                mark_window_pending(runtime_window, "post_present_scene_sync");
                            } else {
                                runtime_window.presentation_pending = false;
                            }
                        }
                        Ok(RenderOutcome::Reconfigure) => {
                            mark_window_pending(runtime_window, "render_reconfigure");
                        }
                        Ok(RenderOutcome::RecreateSurface) => {
                            let _ = render_system.recreate_surface(
                                &mut runtime_window.render_state,
                                runtime_window.native_window.clone(),
                            );
                            mark_window_pending(runtime_window, "render_recreate_surface");
                        }
                        Ok(RenderOutcome::Unavailable) => {
                            runtime_window.presentation_pending = true;
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
        self.process_window_requests(event_loop);
        self.process_runtime_updates();
        self.drive_pending_presentations();
    }
}

fn mark_window_pending(runtime_window: &mut RuntimeWindow, _reason: &'static str) {
    runtime_window.presentation_pending = true;
    if window_can_present(runtime_window) {
        runtime_window.native_window.request_redraw();
    }
}

fn sync_window_metrics(
    runtime_window: &mut RuntimeWindow,
    metrics: WindowMetrics,
    text_system: &mut TextSystem,
    render_system: Option<&mut RenderSystem>,
    sync_scene_now: bool,
) {
    runtime_window.public_window.set_metrics(
        metrics.logical_size,
        metrics.physical_size,
        metrics.scale_factor,
    );
    if let Some(render_system) = render_system {
        let _ = render_system.resize(&mut runtime_window.render_state, metrics.physical_size);
    }
    if sync_scene_now {
        runtime_window
            .retained_tree
            .compute_layout(metrics.logical_size, text_system);
        runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
        runtime_window.metrics_scene_sync_pending = false;
    } else {
        runtime_window.metrics_scene_sync_pending = true;
    }
    mark_window_pending(runtime_window, "sync_window_metrics");
}

fn sync_window_to_native_state(
    runtime_window: &mut RuntimeWindow,
    text_system: &mut TextSystem,
    render_system: Option<&mut RenderSystem>,
) {
    let metrics = current_window_metrics(&runtime_window.native_window);
    if window_metrics_changed(runtime_window, metrics) {
        sync_window_metrics(runtime_window, metrics, text_system, render_system, false);
    }
}

pub(crate) fn run_application(
    last_window_behavior: LastWindowBehavior,
    on_launch: impl FnOnce(&mut App) -> Result<(), Error> + 'static,
) -> Result<(), Error> {
    let event_loop = EventLoop::<RunnerEvent>::with_user_event()
        .build()
        .map_err(|error| PlatformError::new(error.to_string()))?;
    let proxy: EventLoopProxy<RunnerEvent> = event_loop.create_proxy();

    let mut app = App::new();
    app.set_wake_handle(Some(Arc::new(move || {
        let _ = proxy.send_event(RunnerEvent::Wake);
    })));

    on_launch(&mut app)?;

    let mut runtime = AppRuntime::new(app, last_window_behavior);
    event_loop
        .run_app(&mut runtime)
        .map_err(|error| PlatformError::new(error.to_string()).into())
}
