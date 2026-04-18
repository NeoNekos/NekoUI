use std::sync::Arc;

use hashbrown::HashSet;
use winit::dpi::PhysicalSize;
use winit::window::Window as WinitWindow;

use crate::scene::RetainedTree;
use crate::window::{Window, WindowSize};

use super::super::wgpu::WindowRenderState;

pub(crate) struct RuntimeWindow {
    pub(crate) _internal_id: crate::WindowId,
    pub(crate) public_window: Window,
    pub(crate) native_window: Arc<WinitWindow>,
    pub(crate) template_root: crate::AnyElement,
    pub(crate) referenced_views: HashSet<u64>,
    pub(crate) build_scratch: crate::element::SpecArena,
    pub(crate) retained_tree: RetainedTree,
    pub(crate) compiled_scene: crate::scene::CompiledScene,
    pub(crate) render_state: WindowRenderState,
    pub(crate) presentation_pending: bool,
    pub(crate) metrics_scene_sync_pending: bool,
    pub(crate) occluded: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowMetrics {
    pub(crate) logical_size: WindowSize,
    pub(crate) physical_size: WindowSize,
    pub(crate) scale_factor: f64,
}

pub(crate) fn current_window_metrics(window: &WinitWindow) -> WindowMetrics {
    metrics_from_parts(window.inner_size(), window.scale_factor())
}

pub(crate) fn metrics_from_physical_size(
    runtime_window: &RuntimeWindow,
    physical_size: PhysicalSize<u32>,
) -> WindowMetrics {
    metrics_from_parts(physical_size, runtime_window.native_window.scale_factor())
}

pub(crate) fn metrics_from_scale_change(
    runtime_window: &RuntimeWindow,
    scale_factor: f64,
) -> WindowMetrics {
    metrics_from_parts(runtime_window.native_window.inner_size(), scale_factor)
}

pub(crate) fn window_depends_on_dirty_views(
    runtime_window: &RuntimeWindow,
    dirty_views: &hashbrown::HashMap<u64, crate::scene::DirtyLaneMask>,
) -> bool {
    dirty_views
        .keys()
        .any(|view_id| runtime_window.referenced_views.contains(view_id))
}

pub(crate) fn window_metrics_changed(
    runtime_window: &RuntimeWindow,
    metrics: WindowMetrics,
) -> bool {
    runtime_window.public_window.size() != metrics.logical_size
        || runtime_window.public_window.physical_size() != metrics.physical_size
        || (runtime_window.public_window.scale_factor() - metrics.scale_factor).abs() > f64::EPSILON
}

pub(crate) fn window_can_present(runtime_window: &RuntimeWindow) -> bool {
    !runtime_window.occluded
        && !window_is_zero_sized(runtime_window)
        && !runtime_window.native_window.is_minimized().unwrap_or(false)
}

pub(crate) fn window_is_zero_sized(runtime_window: &RuntimeWindow) -> bool {
    let size = runtime_window.public_window.physical_size();
    size.width == 0 || size.height == 0
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

#[cfg(test)]
mod tests {
    use crate::window::WindowSize;

    use super::WindowMetrics;

    #[test]
    fn metrics_difference_detects_zero_and_scale_changes() {
        let old = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(1600, 1200),
            scale_factor: 2.0,
        };
        let new = WindowMetrics {
            logical_size: WindowSize::new(800, 600),
            physical_size: WindowSize::new(0, 0),
            scale_factor: 2.0,
        };
        assert_ne!(old.physical_size, new.physical_size);
    }
}
