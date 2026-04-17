use crate::style::Color;
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

#[derive(Debug, Clone)]
pub struct CompiledScene {
    pub clear_color: Option<Color>,
    pub primitives: Vec<Primitive>,
}

#[derive(Debug, Clone)]
pub enum Primitive {
    Quad {
        bounds: LayoutBox,
        color: Color,
    },
    Text {
        bounds: LayoutBox,
        layout: TextLayout,
        color: Color,
    },
}
