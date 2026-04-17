use std::sync::Arc;

use hashbrown::{HashMap, HashSet};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window as WinitWindow, WindowAttributes, WindowId as WinitWindowId};

use crate::app::{App, LastWindowBehavior};
use crate::error::{Error, PlatformError};
use crate::platform::wgpu::{RenderOutcome, RenderSystem, WindowRenderState};
use crate::scene::RetainedTree;
use crate::text_system::TextSystem;
use crate::window::{Window, WindowSize};

#[derive(Debug, Clone, Copy)]
enum RunnerEvent {
    Wake,
}

struct RuntimeWindow {
    _internal_id: crate::WindowId,
    public_window: Window,
    native_window: Arc<WinitWindow>,
    template_root: crate::Element,
    referenced_views: HashSet<u64>,
    retained_tree: RetainedTree,
    compiled_scene: crate::scene::CompiledScene,
    render_state: WindowRenderState,
}

struct Runner {
    app: App,
    last_window_behavior: LastWindowBehavior,
    windows: HashMap<WinitWindowId, RuntimeWindow>,
    text_system: TextSystem,
    render_system: Option<RenderSystem>,
}

#[derive(Debug, Clone, Copy)]
struct WindowMetrics {
    logical_size: WindowSize,
    physical_size: WindowSize,
    scale_factor: f64,
}

impl Runner {
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
            let (resolved_root, referenced_views) = match self
                .app
                .resolve_root_element_with_views(&mut public_window, &template_root)
            {
                Ok(value) => value,
                Err(error) => {
                    log::error!("failed to resolve root element: {error}");
                    continue;
                }
            };
            let mut retained_tree = RetainedTree::from_element(&resolved_root);

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
                    referenced_views,
                    retained_tree,
                    compiled_scene,
                    render_state,
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

        for runtime_window in self.windows.values_mut() {
            if !window_depends_on_dirty_views(runtime_window, &runtime.dirty_views) {
                continue;
            }

            let (resolved_root, referenced_views) = match self.app.resolve_root_element_with_views(
                &mut runtime_window.public_window,
                &runtime_window.template_root,
            ) {
                Ok(value) => value,
                Err(error) => {
                    log::error!("failed to rebuild root element: {error}");
                    continue;
                }
            };
            runtime_window.referenced_views = referenced_views;
            let dirty = runtime_window
                .retained_tree
                .update_from_element(&resolved_root);

            if dirty.needs_layout() {
                runtime_window
                    .retained_tree
                    .compute_layout(runtime_window.public_window.size(), &mut self.text_system);
            }
            if dirty.needs_scene_compile() {
                runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
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

impl ApplicationHandler<RunnerEvent> for Runner {
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
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    let metrics = metrics_from_physical_size(runtime_window, size);
                    runtime_window.public_window.set_metrics(
                        metrics.logical_size,
                        metrics.physical_size,
                        metrics.scale_factor,
                    );
                    if let Some(render_system) = self.render_system.as_mut() {
                        let _ = render_system
                            .resize(&mut runtime_window.render_state, metrics.physical_size);
                    }
                    runtime_window
                        .retained_tree
                        .compute_layout(metrics.logical_size, &mut self.text_system);
                    runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
                    runtime_window.native_window.request_redraw();
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
                if let Some(runtime_window) = self.windows.get_mut(&window_id) {
                    let metrics = metrics_from_scale_change(runtime_window, scale_factor);
                    runtime_window.public_window.set_metrics(
                        metrics.logical_size,
                        metrics.physical_size,
                        metrics.scale_factor,
                    );
                    if let Some(render_system) = self.render_system.as_mut() {
                        let _ = render_system
                            .resize(&mut runtime_window.render_state, metrics.physical_size);
                    }
                    runtime_window
                        .retained_tree
                        .compute_layout(metrics.logical_size, &mut self.text_system);
                    runtime_window.compiled_scene = runtime_window.retained_tree.compile_scene();
                    runtime_window.native_window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(render_system), Some(runtime_window)) = (
                    self.render_system.as_mut(),
                    self.windows.get_mut(&window_id),
                ) {
                    match render_system.render(
                        &mut runtime_window.render_state,
                        &runtime_window.compiled_scene,
                        &mut self.text_system,
                        &runtime_window.native_window,
                        runtime_window.public_window.scale_factor(),
                    ) {
                        Ok(RenderOutcome::Presented | RenderOutcome::Skip) => {}
                        Ok(RenderOutcome::Reconfigure) => {
                            let _ = render_system.resize(
                                &mut runtime_window.render_state,
                                runtime_window.public_window.physical_size(),
                            );
                            runtime_window.native_window.request_redraw();
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

    let mut runner = Runner::new(app, last_window_behavior);
    event_loop
        .run_app(&mut runner)
        .map_err(|error| PlatformError::new(error.to_string()).into())
}

fn current_window_metrics(window: &WinitWindow) -> WindowMetrics {
    metrics_from_parts(window.inner_size(), window.scale_factor())
}

fn metrics_from_physical_size(
    runtime_window: &RuntimeWindow,
    physical_size: PhysicalSize<u32>,
) -> WindowMetrics {
    metrics_from_parts(physical_size, runtime_window.native_window.scale_factor())
}

fn metrics_from_scale_change(runtime_window: &RuntimeWindow, scale_factor: f64) -> WindowMetrics {
    metrics_from_parts(runtime_window.native_window.inner_size(), scale_factor)
}

fn metrics_from_parts(physical_size: PhysicalSize<u32>, scale_factor: f64) -> WindowMetrics {
    let scale_factor = sanitize_scale_factor(scale_factor);
    let logical_width = (f64::from(physical_size.width) / scale_factor)
        .round()
        .max(1.0) as u32;
    let logical_height = (f64::from(physical_size.height) / scale_factor)
        .round()
        .max(1.0) as u32;

    WindowMetrics {
        logical_size: WindowSize::new(logical_width, logical_height),
        physical_size: WindowSize::new(physical_size.width, physical_size.height),
        scale_factor,
    }
}

fn sanitize_scale_factor(scale_factor: f64) -> f64 {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    }
}

fn window_depends_on_dirty_views(
    runtime_window: &RuntimeWindow,
    dirty_views: &HashMap<u64, crate::scene::DirtyLaneMask>,
) -> bool {
    dirty_views
        .keys()
        .any(|view_id| runtime_window.referenced_views.contains(view_id))
}
