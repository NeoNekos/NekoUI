use std::ops::Range;
use std::sync::Arc;

use crate::style::{Color, CornerRadii, EdgeWidths, LinearGradient};
use crate::text_system::TextLayout;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for LayoutBox {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneNodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PrimitiveRange {
    pub start: u32,
    pub end: u32,
}

#[expect(
    dead_code,
    reason = "Phase 1 scene metadata API is ahead of full renderer usage"
)]
impl PrimitiveRange {
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub fn as_range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Transform2D {
    pub tx: f32,
    pub ty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClipInfo {
    pub bounds: Option<LayoutBox>,
}

pub type EffectMask = u32;

#[derive(Debug, Clone)]
pub struct SceneNode {
    pub parent: Option<SceneNodeId>,
    pub first_child: Option<SceneNodeId>,
    pub next_sibling: Option<SceneNodeId>,
    pub transform: Transform2D,
    pub clip: ClipInfo,
    pub opacity: f32,
    pub effect_mask: EffectMask,
    pub primitive_range: PrimitiveRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialClass {
    Rect,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClipClass {
    #[default]
    None,
    Rect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EffectClass {
    #[default]
    None,
    Opacity,
}

#[derive(Debug, Clone)]
pub struct LogicalBatch {
    pub primitive_range: PrimitiveRange,
    pub material_class: MaterialClass,
    pub clip_class: ClipClass,
    pub effect_class: EffectClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EffectRegionKind {
    #[default]
    None,
}

#[expect(
    dead_code,
    reason = "Effect region metadata is reserved for later scene/render integration"
)]
#[derive(Debug, Clone)]
pub struct EffectRegion {
    pub bounds: LayoutBox,
    pub kind: EffectRegionKind,
}

#[expect(
    dead_code,
    reason = "Compiled scene carries forward-looking metadata for upcoming passes"
)]
#[derive(Debug, Clone)]
pub struct CompiledScene {
    pub clear_color: Option<Color>,
    pub scene_nodes: Arc<[SceneNode]>,
    pub primitives: Arc<[Primitive]>,
    pub logical_batches: Arc<[LogicalBatch]>,
    pub effect_regions: Arc<[EffectRegion]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RectFill {
    Solid(Color),
    LinearGradient(LinearGradient),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectPrimitive {
    pub bounds: LayoutBox,
    pub fill: RectFill,
    pub corner_radii: CornerRadii,
    pub border_widths: EdgeWidths,
    pub border_color: Option<Color>,
    pub opacity: f32,
}

#[derive(Debug, Clone)]
pub enum Primitive {
    Rect(RectPrimitive),
    Text {
        bounds: LayoutBox,
        layout: TextLayout,
        color: Color,
    },
}

impl Primitive {
    pub fn material_class(&self) -> MaterialClass {
        match self {
            Primitive::Rect(_) => MaterialClass::Rect,
            Primitive::Text { .. } => MaterialClass::Text,
        }
    }
}
