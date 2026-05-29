use crate::layout::LayoutRect;
use crate::render::{DrawItemKind, PreparedFrame};
use crate::style::Color;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SolidRectDraw {
    pub(super) rect: LayoutRect,
    pub(super) color: Color,
}

pub(super) fn collect_solid_rects(prepared: &PreparedFrame) -> Vec<SolidRectDraw> {
    prepared
        .draw_items()
        .iter()
        .filter_map(|item| match item.kind() {
            DrawItemKind::Rect { color } if color.srgb_channels().is_some() => {
                Some(SolidRectDraw {
                    rect: item.rect(),
                    color: *color,
                })
            }
            DrawItemKind::Rect { .. } => None,
            DrawItemKind::Text { .. }
            | DrawItemKind::ClipPush
            | DrawItemKind::ClipPop
            | DrawItemKind::Unsupported { .. } => None,
        })
        .collect()
}

pub(super) fn solid_rect_count(prepared: &PreparedFrame) -> usize {
    prepared
        .draw_items()
        .iter()
        .filter(|item| is_supported_solid_rect(item.kind()))
        .count()
}

pub(super) fn unsupported_draw_items(prepared: &PreparedFrame) -> usize {
    prepared
        .draw_items()
        .iter()
        .filter(|item| {
            !is_supported_solid_rect(item.kind()) && !item.kind().supported_windows_glyph_text()
        })
        .count()
}

fn is_supported_solid_rect(kind: &DrawItemKind) -> bool {
    match kind {
        DrawItemKind::Rect { color } => color.srgb_channels().is_some(),
        DrawItemKind::Text { .. }
        | DrawItemKind::ClipPush
        | DrawItemKind::ClipPop
        | DrawItemKind::Unsupported { .. } => false,
    }
}

#[cfg(all(test, target_os = "windows"))]
#[allow(dead_code)]
pub(crate) fn count_unsupported_draw_items_for_backend(prepared: &PreparedFrame) -> usize {
    unsupported_draw_items(prepared)
}
