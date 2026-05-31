use std::time::Duration;

use crate::layout::{LayoutRect, LayoutSize, Viewport, ViewportGeneration};
use crate::platform::PhysicalSize;
use crate::scene::{
    PaintScene, ResourceDemandKind, SceneGeneration, SceneInputSignature, SceneOrder,
};
use crate::style::Color;
use crate::text::{TextGlyphDemand, TextLayoutRef};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PreparedFrameGeneration {
    scene: SceneGeneration,
    surface_generation: Option<u64>,
}

impl PreparedFrameGeneration {
    pub(crate) fn new(scene: SceneGeneration) -> Self {
        Self {
            scene,
            surface_generation: None,
        }
    }

    pub(crate) fn with_surface(scene: SceneGeneration, surface_generation: u64) -> Self {
        Self {
            scene,
            surface_generation: Some(surface_generation),
        }
    }

    pub(crate) fn matches_scene(&self, scene: &SceneGeneration) -> bool {
        &self.scene == scene
    }

    pub(crate) fn matches_scene_and_surface(
        &self,
        scene: &SceneGeneration,
        surface_generation: u64,
    ) -> bool {
        self.matches_scene(scene) && self.surface_generation == Some(surface_generation)
    }

    #[cfg(test)]
    pub(crate) fn scene(&self) -> &SceneGeneration {
        &self.scene
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn surface_generation(&self) -> Option<u64> {
        self.surface_generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PreparedFrameContext {
    logical_viewport_size: LayoutSize,
    physical_surface_size: PhysicalSize,
    scale_factor: f32,
    viewport_generation: ViewportGeneration,
    surface_generation: Option<u64>,
}

impl PreparedFrameContext {
    #[cfg(test)]
    pub(crate) fn scene_only() -> Self {
        Self {
            logical_viewport_size: LayoutSize::new(800.0, 600.0),
            physical_surface_size: PhysicalSize::ZERO,
            scale_factor: 1.0,
            viewport_generation: ViewportGeneration::INITIAL,
            surface_generation: None,
        }
    }

    pub(crate) fn for_surface(
        viewport: Viewport,
        physical_surface_size: PhysicalSize,
        surface_generation: u64,
    ) -> Self {
        Self {
            logical_viewport_size: viewport.logical_size(),
            physical_surface_size,
            scale_factor: viewport.scale_factor(),
            viewport_generation: viewport.generation(),
            surface_generation: Some(surface_generation),
        }
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn logical_viewport_size(self) -> LayoutSize {
        self.logical_viewport_size
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn physical_surface_size(self) -> PhysicalSize {
        self.physical_surface_size
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn scale_factor(self) -> f32 {
        self.scale_factor
    }

    #[cfg(test)]
    pub(crate) fn viewport_generation(self) -> ViewportGeneration {
        self.viewport_generation
    }

    pub(crate) fn surface_generation(self) -> Option<u64> {
        self.surface_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum RenderPass {
    Upload,
    MainColor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PreparedPass {
    class: RenderPass,
    draw_orders: Vec<SceneOrder>,
    upload_count: usize,
}

impl PreparedPass {
    pub(crate) fn new(
        class: RenderPass,
        draw_orders: Vec<SceneOrder>,
        upload_count: usize,
    ) -> Self {
        Self {
            class,
            draw_orders,
            upload_count,
        }
    }

    #[cfg(test)]
    pub(crate) fn class(&self) -> RenderPass {
        self.class
    }

    #[cfg(test)]
    pub(crate) fn draw_orders(&self) -> &[SceneOrder] {
        &self.draw_orders
    }

    #[cfg(test)]
    pub(crate) fn upload_count(&self) -> usize {
        self.upload_count
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DrawItem {
    order: SceneOrder,
    node_id: u64,
    rect: LayoutRect,
    kind: DrawItemKind,
}

impl DrawItem {
    pub(crate) fn new(
        order: SceneOrder,
        node_id: u64,
        rect: LayoutRect,
        kind: DrawItemKind,
    ) -> Self {
        Self {
            order,
            node_id,
            rect,
            kind,
        }
    }

    pub(crate) fn order(&self) -> SceneOrder {
        self.order
    }

    pub(crate) fn node_id(&self) -> u64 {
        self.node_id
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn rect(&self) -> LayoutRect {
        self.rect
    }

    pub(crate) fn kind(&self) -> &DrawItemKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DrawItemKind {
    Rect {
        color: Color,
    },
    Text {
        text_generation: SceneInputSignature,
        text_metrics_generation: u64,
        layout: TextLayoutRef,
        clip: Option<LayoutRect>,
        color: Color,
    },
    ClipPush,
    ClipPop,
    Unsupported {
        capability: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UploadIntent {
    kind: ResourceDemandKind,
    owner_node_id: u64,
    expected_generation: SceneInputSignature,
    dependent_draw_orders: Vec<SceneOrder>,
    glyphs: Option<TextGlyphDemand>,
}

impl UploadIntent {
    pub(crate) fn new(
        kind: ResourceDemandKind,
        owner_node_id: u64,
        expected_generation: SceneInputSignature,
        dependent_draw_orders: Vec<SceneOrder>,
    ) -> Self {
        Self {
            kind,
            owner_node_id,
            expected_generation,
            dependent_draw_orders,
            glyphs: None,
        }
    }

    pub(crate) fn glyph(
        owner_node_id: u64,
        expected_generation: SceneInputSignature,
        dependent_draw_orders: Vec<SceneOrder>,
        glyphs: TextGlyphDemand,
    ) -> Self {
        Self {
            kind: ResourceDemandKind::Glyph,
            owner_node_id,
            expected_generation,
            dependent_draw_orders,
            glyphs: Some(glyphs),
        }
    }

    #[cfg(test)]
    pub(crate) fn kind(&self) -> ResourceDemandKind {
        self.kind
    }

    #[cfg(test)]
    pub(crate) fn owner_node_id(&self) -> u64 {
        self.owner_node_id
    }

    #[cfg(test)]
    pub(crate) fn expected_generation(&self) -> &SceneInputSignature {
        &self.expected_generation
    }

    #[cfg(test)]
    pub(crate) fn dependent_draw_orders(&self) -> &[SceneOrder] {
        &self.dependent_draw_orders
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn glyphs(&self) -> Option<&TextGlyphDemand> {
        self.glyphs.as_ref()
    }
}

impl DrawItemKind {
    #[cfg(target_os = "windows")]
    pub(crate) fn supported_windows_glyph_text(&self) -> bool {
        match self {
            Self::Text { layout, color, .. } => {
                color.srgb_channels().is_some() && !layout.glyphs().is_empty()
            }
            Self::Rect { .. } | Self::ClipPush | Self::ClipPop | Self::Unsupported { .. } => false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct UploadPlan {
    intents: Vec<UploadIntent>,
}

impl UploadPlan {
    pub(crate) fn new(intents: Vec<UploadIntent>) -> Self {
        Self { intents }
    }

    pub(crate) fn intents(&self) -> &[UploadIntent] {
        &self.intents
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameGraphStats {
    pub surface_generation: Option<u64>,
    pub pass_count: usize,
    pub draw_item_count: usize,
    pub upload_intent_count: usize,
    pub layer_count: usize,
    pub unsupported_fragment_count: usize,
    pub stale_drop_count: u64,
    pub duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PreparedFrame {
    generation: PreparedFrameGeneration,
    context: PreparedFrameContext,
    upload_plan: UploadPlan,
    passes: Vec<PreparedPass>,
    draw_items: Vec<DrawItem>,
    stats: FrameGraphStats,
}

impl PreparedFrame {
    pub(crate) fn new(
        generation: PreparedFrameGeneration,
        context: PreparedFrameContext,
        upload_plan: UploadPlan,
        passes: Vec<PreparedPass>,
        draw_items: Vec<DrawItem>,
        stats: FrameGraphStats,
    ) -> Self {
        Self {
            generation,
            context,
            upload_plan,
            passes,
            draw_items,
            stats,
        }
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn generation(&self) -> &PreparedFrameGeneration {
        &self.generation
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn context(&self) -> PreparedFrameContext {
        self.context
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn upload_plan(&self) -> &UploadPlan {
        &self.upload_plan
    }

    #[cfg(test)]
    pub(crate) fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn draw_items(&self) -> &[DrawItem] {
        &self.draw_items
    }

    pub(crate) fn stats(&self) -> &FrameGraphStats {
        &self.stats
    }

    pub(crate) fn is_current_for_scene_and_surface(
        &self,
        scene: &PaintScene,
        surface_generation: u64,
    ) -> bool {
        self.generation
            .matches_scene_and_surface(&scene.generation(), surface_generation)
    }

    #[cfg(test)]
    pub(crate) fn is_current_for_scene(&self, scene: &PaintScene) -> bool {
        self.generation.matches_scene(&scene.generation())
    }
}
