use crate::layout::LayoutRect;
use crate::render::{DrawItemKind, PreparedFrame};
use crate::scene::BoxShape;
use crate::style::{Color, CornerRadii, Edges, Length};

use super::clip::{ActiveClip, ClipStack};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BoxShapeDraw {
    pub(super) rect: LayoutRect,
    pub(super) shape: BoxShape,
}

pub(super) fn collect_box_shapes(prepared: &PreparedFrame) -> Vec<BoxShapeDraw> {
    let mut shapes = Vec::with_capacity(prepared.draw_items().len());
    for item in prepared.draw_items() {
        let shape = match item.kind() {
            DrawItemKind::BoxShape { shape } if supported_box_shape(*shape) => Some(BoxShapeDraw {
                rect: item.rect(),
                shape: *shape,
            }),
            DrawItemKind::Rect { color } if color.to_current_backend_sdr_srgb_rgba().is_some() => {
                Some(BoxShapeDraw {
                    rect: item.rect(),
                    shape: BoxShape::new(
                        Some(*color),
                        None,
                        Edges::all(Length::ZERO),
                        CornerRadii::all(Length::ZERO),
                        crate::style::Opacity::OPAQUE,
                    ),
                })
            }
            DrawItemKind::BoxShape { .. } | DrawItemKind::Rect { .. } => None,
            DrawItemKind::Text { .. }
            | DrawItemKind::ClipPush { .. }
            | DrawItemKind::ClipPop
            | DrawItemKind::Unsupported { .. } => None,
        };
        if let Some(shape) = shape {
            shapes.push(shape);
        }
    }
    shapes
}

pub(super) fn box_shape_count(prepared: &PreparedFrame) -> usize {
    let mut count = 0;
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            kind => {
                if clip_stack.active_clip() != ActiveClip::Empty && is_supported_box_shape(kind) {
                    count += 1;
                }
            }
        }
    }
    count
}

pub(super) fn unsupported_draw_items(prepared: &PreparedFrame) -> usize {
    let mut unsupported = 0;
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            kind => {
                if clip_stack.active_clip() != ActiveClip::Empty
                    && !is_backend_supported_control_or_draw(kind)
                {
                    unsupported += 1;
                }
            }
        }
    }
    unsupported
}

pub(super) fn supported_box_shape(shape: BoxShape) -> bool {
    unsupported_box_shape_capability(shape).is_none()
}

pub(super) fn unsupported_box_shape_capability(shape: BoxShape) -> Option<&'static str> {
    if !color_supported(shape.fill()) {
        return Some("box_shape.fill.color_space");
    }
    if !color_supported(shape.border_color()) {
        return Some("box_shape.border.color_space");
    }
    None
}

fn is_supported_box_shape(kind: &DrawItemKind) -> bool {
    match kind {
        DrawItemKind::BoxShape { shape } => supported_box_shape(*shape),
        DrawItemKind::Rect { color } => color.to_current_backend_sdr_srgb_rgba().is_some(),
        DrawItemKind::Text { .. }
        | DrawItemKind::ClipPush { .. }
        | DrawItemKind::ClipPop
        | DrawItemKind::Unsupported { .. } => false,
    }
}

fn is_backend_supported_control_or_draw(kind: &DrawItemKind) -> bool {
    is_supported_box_shape(kind)
        || kind.supported_windows_glyph_text()
        || matches!(kind, DrawItemKind::ClipPush { .. } | DrawItemKind::ClipPop)
}

fn color_supported(color: Option<Color>) -> bool {
    match color {
        Some(color) => color.to_current_backend_sdr_srgb_rgba().is_some(),
        None => true,
    }
}

#[cfg(all(test, target_os = "windows"))]
#[allow(dead_code)]
pub(crate) fn count_unsupported_draw_items_for_backend(prepared: &PreparedFrame) -> usize {
    unsupported_draw_items(prepared)
}
