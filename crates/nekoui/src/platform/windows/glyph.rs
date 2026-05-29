use std::collections::HashMap;

use crate::error::{NekoError, NekoResult};
use crate::layout::LayoutRect;
use crate::render::{DrawItemKind, PreparedFrame};
use crate::style::Color;
use crate::text::{
    FontManager, GlyphBitmap, GlyphInstance, GlyphKey, GlyphRasterError, TextLayoutRef,
};

pub(super) const GLYPH_ATLAS_WIDTH: u32 = 1024;
pub(super) const GLYPH_ATLAS_HEIGHT: u32 = 1024;
const GLYPH_ATLAS_PADDING: u32 = 1;
const GLYPH_ATLAS_SIDE_CACHE_LIMIT: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GlyphDraw {
    pub(super) order: crate::scene::SceneOrder,
    pub(super) rect: LayoutRect,
    pub(super) uv: GlyphUv,
    pub(super) color: Color,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GlyphUv {
    pub(super) left: f32,
    pub(super) top: f32,
    pub(super) right: f32,
    pub(super) bottom: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GlyphAtlasEntry {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlyphSkipReason {
    MissingGlyph,
    UnsupportedContent(&'static str),
    AtlasFull,
    ExceedsAtlasPage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlyphAtlasOutcome {
    Ready(GlyphAtlasEntry),
    Unsupported(GlyphSkipReason),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GlyphUnsupportedReport {
    pub(super) missing_glyph_demands: usize,
    pub(super) unsupported_content_demands: usize,
    pub(super) atlas_full_demands: usize,
    pub(super) atlas_oversize_demands: usize,
    skipped_cached_glyph_instances: usize,
    pub(super) missing_atlas_entries: usize,
}

impl GlyphUnsupportedReport {
    fn note_skip(&mut self, reason: GlyphSkipReason) {
        match reason {
            GlyphSkipReason::MissingGlyph => self.missing_glyph_demands += 1,
            GlyphSkipReason::UnsupportedContent(_) => self.unsupported_content_demands += 1,
            GlyphSkipReason::AtlasFull => self.atlas_full_demands += 1,
            GlyphSkipReason::ExceedsAtlasPage => self.atlas_oversize_demands += 1,
        }
    }

    pub(super) fn add(&mut self, other: Self) {
        self.missing_glyph_demands += other.missing_glyph_demands;
        self.unsupported_content_demands += other.unsupported_content_demands;
        self.atlas_full_demands += other.atlas_full_demands;
        self.atlas_oversize_demands += other.atlas_oversize_demands;
        self.skipped_cached_glyph_instances += other.skipped_cached_glyph_instances;
        self.missing_atlas_entries += other.missing_atlas_entries;
    }

    pub(super) const fn is_empty(self) -> bool {
        self.missing_glyph_demands == 0
            && self.unsupported_content_demands == 0
            && self.atlas_full_demands == 0
            && self.atlas_oversize_demands == 0
            && self.skipped_cached_glyph_instances == 0
            && self.missing_atlas_entries == 0
    }

    pub(super) const fn skipped_glyph_instances(self) -> usize {
        self.skipped_cached_glyph_instances + self.missing_atlas_entries
    }
}

impl GlyphAtlasEntry {
    fn uv(self, atlas_width: u32, atlas_height: u32) -> NekoResult<GlyphUv> {
        if atlas_width == 0 || atlas_height == 0 {
            return Err(NekoError::resource_failure(
                "glyph atlas dimensions cannot be zero",
            ));
        }
        let right = checked_u32_add(self.x, self.width, "glyph atlas uv x overflow")?;
        let bottom = checked_u32_add(self.y, self.height, "glyph atlas uv y overflow")?;
        if right > atlas_width || bottom > atlas_height {
            return Err(NekoError::resource_failure(
                "glyph atlas entry exceeds atlas bounds",
            ));
        }
        Ok(GlyphUv {
            left: self.x as f32 / atlas_width as f32,
            top: self.y as f32 / atlas_height as f32,
            right: right as f32 / atlas_width as f32,
            bottom: bottom as f32 / atlas_height as f32,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct GlyphAtlas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    entries: HashMap<GlyphKey, GlyphAtlasEntry>,
    skipped: HashMap<GlyphKey, GlyphSkipReason>,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    dirty: bool,
}

impl GlyphAtlas {
    pub(super) fn new(width: u32, height: u32) -> NekoResult<Self> {
        let len = width
            .checked_mul(height)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| NekoError::resource_failure("glyph atlas page is too large"))?;
        Ok(Self {
            width,
            height,
            pixels: vec![0; len],
            entries: HashMap::new(),
            skipped: HashMap::new(),
            cursor_x: GLYPH_ATLAS_PADDING,
            cursor_y: GLYPH_ATLAS_PADDING,
            row_height: 0,
            dirty: true,
        })
    }

    pub(super) fn ensure_glyph(
        &mut self,
        key: GlyphKey,
        rasterize: impl FnOnce(GlyphKey) -> Result<GlyphBitmap, crate::text::GlyphRasterError>,
    ) -> NekoResult<GlyphAtlasOutcome> {
        if let Some(entry) = self.entries.get(&key).copied() {
            return Ok(GlyphAtlasOutcome::Ready(entry));
        }
        if let Some(reason) = self.skipped.get(&key).copied() {
            return Ok(GlyphAtlasOutcome::Unsupported(reason));
        }
        let bitmap = match rasterize(key) {
            Ok(bitmap) => bitmap,
            Err(GlyphRasterError::MissingGlyph) => {
                self.remember_skip(key, GlyphSkipReason::MissingGlyph);
                return Ok(GlyphAtlasOutcome::Unsupported(
                    GlyphSkipReason::MissingGlyph,
                ));
            }
            Err(GlyphRasterError::UnsupportedContent(kind)) => {
                self.remember_skip(key, GlyphSkipReason::UnsupportedContent(kind));
                return Ok(GlyphAtlasOutcome::Unsupported(
                    GlyphSkipReason::UnsupportedContent(kind),
                ));
            }
        };
        if let Some(reason) = bitmap_skip_reason(&bitmap)? {
            self.remember_skip(key, reason);
            return Ok(GlyphAtlasOutcome::Unsupported(reason));
        }
        let entry = match self.allocate(&bitmap)? {
            GlyphAllocation::Ready(entry) => entry,
            GlyphAllocation::Unsupported(reason) => {
                self.remember_skip(key, reason);
                return Ok(GlyphAtlasOutcome::Unsupported(reason));
            }
        };
        self.copy_bitmap(&bitmap, entry)?;
        self.entries.insert(key, entry);
        self.dirty = true;
        Ok(GlyphAtlasOutcome::Ready(entry))
    }

    pub(super) fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub(super) fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    fn remember_skip(&mut self, key: GlyphKey, reason: GlyphSkipReason) {
        if reason == GlyphSkipReason::AtlasFull {
            return;
        }
        if self.skipped.len() < GLYPH_ATLAS_SIDE_CACHE_LIMIT {
            self.skipped.insert(key, reason);
        }
    }

    #[cfg(test)]
    pub(super) fn pixel(&self, x: u32, y: u32) -> u8 {
        let offset = y
            .checked_mul(self.width)
            .and_then(|value| value.checked_add(x))
            .and_then(|value| usize::try_from(value).ok())
            .expect("test glyph atlas pixel coordinate is out of bounds");
        self.pixels[offset]
    }

    fn allocate(&mut self, bitmap: &GlyphBitmap) -> NekoResult<GlyphAllocation> {
        let atlas_padding = GLYPH_ATLAS_PADDING
            .checked_mul(2)
            .ok_or_else(|| NekoError::resource_failure("glyph atlas padding overflow"))?;
        let padded_width = bitmap
            .width()
            .checked_add(atlas_padding)
            .ok_or_else(|| NekoError::resource_failure("glyph atlas allocation width overflow"))?;
        let padded_height = bitmap
            .height()
            .checked_add(atlas_padding)
            .ok_or_else(|| NekoError::resource_failure("glyph atlas allocation height overflow"))?;
        if padded_width > self.width || padded_height > self.height {
            return Ok(GlyphAllocation::Unsupported(
                GlyphSkipReason::ExceedsAtlasPage,
            ));
        }
        let mut cursor_x = self.cursor_x;
        let mut cursor_y = self.cursor_y;
        let mut row_height = self.row_height;
        if u32_add_exceeds_bound(cursor_x, padded_width, self.width) {
            cursor_x = GLYPH_ATLAS_PADDING;
            let Some(next_cursor_y) = cursor_y
                .checked_add(row_height)
                .and_then(|value| value.checked_add(GLYPH_ATLAS_PADDING))
            else {
                return Ok(GlyphAllocation::Unsupported(GlyphSkipReason::AtlasFull));
            };
            cursor_y = next_cursor_y;
            row_height = 0;
        }
        if u32_add_exceeds_bound(cursor_y, padded_height, self.height) {
            return Ok(GlyphAllocation::Unsupported(GlyphSkipReason::AtlasFull));
        }
        let entry_x = checked_u32_add(
            cursor_x,
            GLYPH_ATLAS_PADDING,
            "glyph atlas entry x overflow",
        )?;
        let entry_y = checked_u32_add(
            cursor_y,
            GLYPH_ATLAS_PADDING,
            "glyph atlas entry y overflow",
        )?;
        let next_cursor_x = checked_u32_add(
            cursor_x,
            padded_width,
            "glyph atlas cursor advance overflow",
        )?;
        let entry = GlyphAtlasEntry {
            x: entry_x,
            y: entry_y,
            width: bitmap.width(),
            height: bitmap.height(),
            left: bitmap.left(),
            top: bitmap.top(),
        };
        self.cursor_x = next_cursor_x;
        self.cursor_y = cursor_y;
        self.row_height = row_height.max(padded_height);
        Ok(GlyphAllocation::Ready(entry))
    }

    fn copy_bitmap(&mut self, bitmap: &GlyphBitmap, entry: GlyphAtlasEntry) -> NekoResult<()> {
        let width = usize::try_from(bitmap.width())
            .map_err(|_| NekoError::resource_failure("glyph bitmap width exceeds usize"))?;
        let height = usize::try_from(bitmap.height())
            .map_err(|_| NekoError::resource_failure("glyph bitmap height exceeds usize"))?;
        let expected_len =
            checked_usize_mul(width, height, "glyph bitmap mask size overflows usize")?;
        if bitmap.pixels().len() != expected_len {
            return Err(NekoError::resource_failure(
                "glyph bitmap mask size does not match placement",
            ));
        }
        let entry_right =
            checked_u32_add(entry.x, bitmap.width(), "glyph atlas entry copy x overflow")?;
        let entry_bottom = checked_u32_add(
            entry.y,
            bitmap.height(),
            "glyph atlas entry copy y overflow",
        )?;
        if entry_right > self.width || entry_bottom > self.height {
            return Err(NekoError::resource_failure(
                "glyph bitmap placement exceeds atlas bounds",
            ));
        }
        let atlas_width = usize::try_from(self.width)
            .map_err(|_| NekoError::resource_failure("glyph atlas width exceeds usize"))?;
        let entry_x = usize::try_from(entry.x)
            .map_err(|_| NekoError::resource_failure("glyph atlas entry x exceeds usize"))?;
        let entry_y = usize::try_from(entry.y)
            .map_err(|_| NekoError::resource_failure("glyph atlas entry y exceeds usize"))?;
        for row in 0..height {
            let dst_y = checked_usize_add(entry_y, row, "glyph bitmap destination y overflow")?;
            let dst_row_start =
                checked_usize_mul(dst_y, atlas_width, "glyph bitmap destination row overflow")?;
            let dst_start = checked_usize_add(
                dst_row_start,
                entry_x,
                "glyph bitmap destination start overflow",
            )?;
            let dst_end =
                checked_usize_add(dst_start, width, "glyph bitmap destination end overflow")?;
            let src_start = checked_usize_mul(row, width, "glyph bitmap source row overflow")?;
            let src_end = checked_usize_add(src_start, width, "glyph bitmap source end overflow")?;
            let source = bitmap.pixels().get(src_start..src_end).ok_or_else(|| {
                NekoError::resource_failure("glyph bitmap mask size does not match placement")
            })?;
            let destination = self.pixels.get_mut(dst_start..dst_end).ok_or_else(|| {
                NekoError::resource_failure("glyph bitmap placement exceeds atlas bounds")
            })?;
            destination.copy_from_slice(source);
        }
        Ok(())
    }
}

fn checked_u32_add(left: u32, right: u32, message: &'static str) -> NekoResult<u32> {
    left.checked_add(right)
        .ok_or_else(|| NekoError::resource_failure(message))
}

fn checked_usize_add(left: usize, right: usize, message: &'static str) -> NekoResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| NekoError::resource_failure(message))
}

fn checked_usize_mul(left: usize, right: usize, message: &'static str) -> NekoResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| NekoError::resource_failure(message))
}

fn u32_add_exceeds_bound(left: u32, right: u32, bound: u32) -> bool {
    left.checked_add(right).is_none_or(|sum| sum > bound)
}

fn bitmap_skip_reason(bitmap: &GlyphBitmap) -> NekoResult<Option<GlyphSkipReason>> {
    if bitmap.is_empty() {
        return Ok(Some(GlyphSkipReason::UnsupportedContent(
            "empty_glyph_bitmap",
        )));
    }
    let width = usize::try_from(bitmap.width())
        .map_err(|_| NekoError::resource_failure("glyph bitmap width exceeds usize"))?;
    let height = usize::try_from(bitmap.height())
        .map_err(|_| NekoError::resource_failure("glyph bitmap height exceeds usize"))?;
    let expected_len = checked_usize_mul(width, height, "glyph bitmap mask size overflows usize")?;
    if bitmap.pixels().len() != expected_len {
        return Ok(Some(GlyphSkipReason::UnsupportedContent(
            "malformed_glyph_bitmap",
        )));
    }
    Ok(None)
}

enum GlyphAllocation {
    Ready(GlyphAtlasEntry),
    Unsupported(GlyphSkipReason),
}

pub(super) fn prepare_glyph_atlas(
    prepared: &PreparedFrame,
    atlas: &mut GlyphAtlas,
    font_manager: &FontManager,
) -> NekoResult<GlyphUnsupportedReport> {
    let mut report = GlyphUnsupportedReport::default();
    for intent in prepared.upload_plan().intents() {
        let Some(glyphs) = intent.glyphs() else {
            continue;
        };
        for demand in glyphs.layout().glyph_demands() {
            if let GlyphAtlasOutcome::Unsupported(reason) =
                atlas.ensure_glyph(demand.key(), |key| font_manager.rasterize_glyph(key))?
            {
                report.note_skip(reason);
            }
        }
    }
    Ok(report)
}

pub(super) fn collect_glyph_draws(
    prepared: &PreparedFrame,
    atlas: &GlyphAtlas,
    draws: &mut Vec<GlyphDraw>,
) -> NekoResult<GlyphUnsupportedReport> {
    draws.clear();
    let mut unsupported = GlyphUnsupportedReport::default();
    for item in prepared.draw_items() {
        let DrawItemKind::Text { layout, color, .. } = item.kind() else {
            continue;
        };
        if color.srgb_channels().is_none() {
            continue;
        }
        push_layout_draws(
            draws,
            &mut unsupported,
            item.order(),
            item.rect(),
            layout,
            *color,
            atlas,
        )?;
    }
    Ok(unsupported)
}

fn push_layout_draws(
    draws: &mut Vec<GlyphDraw>,
    unsupported: &mut GlyphUnsupportedReport,
    order: crate::scene::SceneOrder,
    origin: LayoutRect,
    layout: &TextLayoutRef,
    color: Color,
    atlas: &GlyphAtlas,
) -> NekoResult<()> {
    let scale_factor = layout.scale_factor();
    for glyph in layout.glyphs() {
        if let Some(reason) = atlas.skipped.get(&glyph.key()).copied() {
            unsupported.note_skip(reason);
            unsupported.skipped_cached_glyph_instances += 1;
            continue;
        }
        let Some(entry) = atlas.entries.get(&glyph.key()).copied() else {
            unsupported.missing_atlas_entries += 1;
            continue;
        };
        if entry.width == 0 || entry.height == 0 {
            continue;
        }
        draws.push(GlyphDraw {
            order,
            rect: glyph_rect(origin, *glyph, entry, scale_factor),
            uv: entry.uv(atlas.width, atlas.height)?,
            color,
        });
    }
    Ok(())
}

fn glyph_rect(
    origin: LayoutRect,
    glyph: GlyphInstance,
    entry: GlyphAtlasEntry,
    scale_factor: f32,
) -> LayoutRect {
    let inv_scale = 1.0 / scale_factor;
    LayoutRect::new(
        origin.x() + (glyph.x() + entry.left) as f32 * inv_scale,
        origin.y() + (glyph.y() - entry.top) as f32 * inv_scale,
        entry.width as f32 * inv_scale,
        entry.height as f32 * inv_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{Context, Render};
    use crate::element::{Element, IntoElement, text};
    use crate::retained::{NodeGeneration, RetainedNodeId};
    use crate::style::{ResolvedStyle, StyleDeclaration, StyleExt, px};
    use crate::text::{GlyphBitmap, GlyphRasterError};
    use crate::text::{TextGeneration, TextLayoutResult, TextMeasureQuery, TextMeasureSession};
    use crate::window::WindowOptions;

    #[derive(Debug)]
    struct TestRoot {
        root: Element,
    }

    impl Render for TestRoot {
        fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
            self.root.clone()
        }
    }

    #[test]
    fn atlas_starts_zero_filled_and_preserves_padding() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        assert!(atlas.pixels().iter().all(|pixel| *pixel == 0));
        let key = GlyphKey::new(
            cosmic_text::CacheKey::new(
                fontdb::ID::dummy(),
                1,
                12.0,
                (0.0, 0.0),
                fontdb::Weight::NORMAL,
                cosmic_text::CacheKeyFlags::empty(),
            )
            .0,
            1.0,
        );
        let bitmap = GlyphBitmap::new(key, 2, 2, 0, 0, std::sync::Arc::from([7_u8, 8, 9, 10]));
        let entry = match atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };

        assert_eq!(entry.x, 2);
        assert_eq!(entry.y, 2);
        assert_eq!(atlas.pixel(1, 1), 0);
        assert_eq!(atlas.pixel(2, 2), 7);
        assert_eq!(atlas.pixel(3, 3), 10);
        assert_eq!(atlas.pixel(4, 4), 0);
        assert_eq!(entry.uv(8, 8).unwrap().left, 0.25);
        assert_eq!(entry.uv(8, 8).unwrap().right, 0.5);
    }

    #[test]
    fn cached_glyph_hit_reuses_entry_without_rasterizing_or_dirtying_pixels() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 2, 2, 0, 0, std::sync::Arc::from([1_u8, 2, 3, 4]));
        let first = match atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };
        assert!(atlas.take_dirty());
        let pixels = atlas.pixels().to_vec();

        let second = atlas
            .ensure_glyph(key, |_| panic!("cached glyph should not rasterize"))
            .unwrap();

        assert_eq!(second, GlyphAtlasOutcome::Ready(first));
        assert_eq!(atlas.pixels(), pixels.as_slice());
        assert!(!atlas.take_dirty());
    }

    #[test]
    fn font_manager_rasterizes_shaped_glyph_to_real_mask() {
        let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(18.0)), None);
        let font_manager = FontManager::default();
        let mut session = TextMeasureSession::new(&font_manager);
        let layout = match session.layout(TextMeasureQuery {
            node_id: RetainedNodeId::new(1),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text: "A",
            style: style.text(),
            available_inline_width: None,
            font_generation: session.font_generation(),
            scale_generation: 1,
            scale_factor: 1.0,
        }) {
            TextLayoutResult::Ready(layout) => layout,
            TextLayoutResult::Deferred(dependency) => {
                panic!("text layout deferred: {dependency:?}")
            }
            TextLayoutResult::Failed(error) => panic!("text layout failed: {error:?}"),
        };
        let bitmap = layout
            .glyph_demands()
            .iter()
            .copied()
            .find_map(|demand| match font_manager.rasterize_glyph(demand.key()) {
                Ok(bitmap) if !bitmap.is_empty() => Some(bitmap),
                Ok(_) | Err(GlyphRasterError::MissingGlyph) => None,
                Err(error) => panic!("glyph rasterization failed: {error:?}"),
            })
            .expect("at least one shaped glyph should rasterize to a non-empty mask");

        assert_eq!(
            bitmap.pixels().len(),
            bitmap.width() as usize * bitmap.height() as usize
        );
        assert!(bitmap.pixels().iter().any(|pixel| *pixel > 0));
    }

    #[test]
    fn text_to_prepared_frame_rasterizes_atlas_and_collects_glyph_draws() {
        let mut runtime = crate::runtime::Runtime::new();
        let window = runtime
            .open_window(WindowOptions::new(), |_| TestRoot {
                root: text("A").font_size(px(18.0)).into_element(),
            })
            .unwrap();
        let prepared = runtime
            .state()
            .prepared_frame_snapshot(window.id())
            .expect("opening a text window should publish a prepared frame");
        let mut atlas = GlyphAtlas::new(64, 64).unwrap();

        prepare_glyph_atlas(&prepared, &mut atlas, runtime.state().font_manager()).unwrap();
        let mut draws = Vec::new();
        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(!draws.is_empty());
        assert!(unsupported.is_empty());
        assert!(draws.iter().all(|draw| draw.rect.width() > 0.0));
        assert!(draws.iter().all(|draw| draw.rect.height() > 0.0));
        assert!(atlas.pixels().iter().any(|pixel| *pixel > 0));
    }

    #[test]
    fn missing_and_unsupported_glyph_rasterization_are_nonfatal_skips() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let missing_key = test_glyph_key(1);
        let unsupported_key = test_glyph_key(2);

        let missing = atlas
            .ensure_glyph(missing_key, |_| Err(GlyphRasterError::MissingGlyph))
            .unwrap();
        let unsupported = atlas
            .ensure_glyph(unsupported_key, |_| {
                Err(GlyphRasterError::UnsupportedContent("color_glyph"))
            })
            .unwrap();

        assert_eq!(
            missing,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::MissingGlyph)
        );
        assert_eq!(
            unsupported,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent("color_glyph"))
        );
        assert_eq!(
            atlas
                .ensure_glyph(missing_key, |_| panic!("missing glyph should be cached"))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::MissingGlyph)
        );
        assert_eq!(
            atlas
                .ensure_glyph(unsupported_key, |_| panic!(
                    "unsupported glyph should be cached"
                ))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent("color_glyph"))
        );
        assert!(atlas.pixels().iter().all(|pixel| *pixel == 0));
    }

    #[test]
    fn atlas_full_reports_unsupported_without_poisoning_existing_glyphs() {
        let mut atlas = GlyphAtlas::new(4, 4).unwrap();
        let first_key = test_glyph_key(1);
        let second_key = test_glyph_key(2);
        let first = GlyphBitmap::new(first_key, 1, 1, 0, 0, std::sync::Arc::from([11_u8]));
        let second = GlyphBitmap::new(second_key, 1, 1, 0, 0, std::sync::Arc::from([22_u8]));

        assert!(matches!(
            atlas.ensure_glyph(first_key, |_| Ok(first)).unwrap(),
            GlyphAtlasOutcome::Ready(_)
        ));
        assert!(atlas.take_dirty());
        let pixels = atlas.pixels().to_vec();
        let full = atlas.ensure_glyph(second_key, |_| Ok(second)).unwrap();

        assert_eq!(
            full,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::AtlasFull)
        );
        assert_eq!(atlas.pixels(), pixels.as_slice());
        assert!(!atlas.take_dirty());
        assert!(!atlas.skipped.contains_key(&second_key));
        assert!(matches!(
            atlas.ensure_glyph(first_key, |_| unreachable!()).unwrap(),
            GlyphAtlasOutcome::Ready(_)
        ));
    }

    #[test]
    fn atlas_full_is_not_memoized_for_arbitrary_new_keys() {
        let mut atlas = GlyphAtlas::new(4, 4).unwrap();
        let first_key = test_glyph_key(1);
        let first = GlyphBitmap::new(first_key, 1, 1, 0, 0, std::sync::Arc::from([11_u8]));
        assert!(matches!(
            atlas.ensure_glyph(first_key, |_| Ok(first)).unwrap(),
            GlyphAtlasOutcome::Ready(_)
        ));

        let retry_key = test_glyph_key(2);
        let mut retry_rasterize_count = 0;
        for _ in 0..2 {
            let retry = GlyphBitmap::new(retry_key, 1, 1, 0, 0, std::sync::Arc::from([22_u8]));
            assert_eq!(
                atlas
                    .ensure_glyph(retry_key, |_| {
                        retry_rasterize_count += 1;
                        Ok(retry)
                    })
                    .unwrap(),
                GlyphAtlasOutcome::Unsupported(GlyphSkipReason::AtlasFull)
            );
        }

        for glyph_id in 3..32 {
            let key = test_glyph_key(glyph_id);
            let bitmap = GlyphBitmap::new(key, 1, 1, 0, 0, std::sync::Arc::from([33_u8]));
            assert_eq!(
                atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap(),
                GlyphAtlasOutcome::Unsupported(GlyphSkipReason::AtlasFull)
            );
        }

        assert_eq!(retry_rasterize_count, 2);
        assert!(atlas.skipped.is_empty());
    }

    #[test]
    fn oversize_glyph_reports_unsupported_without_corrupting_existing_atlas() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let first_key = test_glyph_key(1);
        let oversize_key = test_glyph_key(2);
        let first = GlyphBitmap::new(first_key, 2, 2, 0, 0, std::sync::Arc::from([5_u8; 4]));
        let oversize = GlyphBitmap::new(oversize_key, 7, 1, 0, 0, std::sync::Arc::from([9_u8; 7]));
        let first_entry = match atlas.ensure_glyph(first_key, |_| Ok(first)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };
        assert!(atlas.take_dirty());
        let pixels = atlas.pixels().to_vec();

        let outcome = atlas.ensure_glyph(oversize_key, |_| Ok(oversize)).unwrap();

        assert_eq!(
            outcome,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::ExceedsAtlasPage)
        );
        assert_eq!(atlas.pixels(), pixels.as_slice());
        assert!(!atlas.take_dirty());
        assert_eq!(
            atlas
                .ensure_glyph(first_key, |_| panic!("existing glyph should stay cached"))
                .unwrap(),
            GlyphAtlasOutcome::Ready(first_entry)
        );
        assert_eq!(
            atlas
                .ensure_glyph(oversize_key, |_| panic!("oversize glyph should be cached"))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::ExceedsAtlasPage)
        );
    }

    #[test]
    fn empty_glyph_bitmaps_are_cached_as_unsupported_nonfatal_skips() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let key = test_glyph_key(1);
        let empty = GlyphBitmap::new(key, 0, 0, 0, 0, std::sync::Arc::from([]));
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );

        assert_eq!(
            atlas.ensure_glyph(key, |_| Ok(empty)).unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "empty_glyph_bitmap"
            ))
        );
        assert_eq!(
            atlas
                .ensure_glyph(key, |_| panic!("empty glyph skip should be cached"))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "empty_glyph_bitmap"
            ))
        );
        let prepared = prepared_frame_for_layout(layout);
        let mut draws = Vec::new();
        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(draws.is_empty());
        assert_eq!(unsupported.unsupported_content_demands, 1);
        assert_eq!(unsupported.missing_atlas_entries, 0);
        assert_eq!(unsupported.skipped_glyph_instances(), 1);
    }

    #[test]
    fn skipped_side_cache_is_bounded() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();

        for index in 0..GLYPH_ATLAS_SIDE_CACHE_LIMIT + 8 {
            let key = test_glyph_key(index as u16);
            let empty = GlyphBitmap::new(key, 0, 0, 0, 0, std::sync::Arc::from([]));
            assert_eq!(
                atlas.ensure_glyph(key, |_| Ok(empty)).unwrap(),
                GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                    "empty_glyph_bitmap"
                ))
            );
        }

        assert_eq!(atlas.skipped.len(), GLYPH_ATLAS_SIDE_CACHE_LIMIT);
    }

    #[test]
    fn malformed_nonzero_glyph_bitmaps_are_cached_as_unsupported_nonfatal_skips() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let key = test_glyph_key(1);
        let malformed = GlyphBitmap::new(key, 2, 2, 0, 0, std::sync::Arc::from([1_u8, 2, 3]));

        assert_eq!(
            atlas.ensure_glyph(key, |_| Ok(malformed)).unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "malformed_glyph_bitmap"
            ))
        );

        assert!(atlas.pixels().iter().all(|pixel| *pixel == 0));
        assert_eq!(
            atlas
                .ensure_glyph(key, |_| panic!("malformed glyph skip should be cached"))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "malformed_glyph_bitmap"
            ))
        );
    }

    #[test]
    fn zero_ink_nonzero_glyph_bitmaps_are_legitimate_atlas_entries() {
        let mut atlas = GlyphAtlas::new(8, 8).unwrap();
        let key = test_glyph_key(1);
        let zero_ink = GlyphBitmap::new(key, 2, 2, 0, 0, std::sync::Arc::from([0_u8; 4]));

        let entry = match atlas.ensure_glyph(key, |_| Ok(zero_ink)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };

        assert_eq!(entry.width, 2);
        assert_eq!(entry.height, 2);
        assert!(atlas.pixels().iter().all(|pixel| *pixel == 0));
    }

    #[test]
    fn glyph_draw_rects_convert_physical_bitmap_bounds_to_logical_space() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key_at_scale(1, 2.0);
        let bitmap = GlyphBitmap::new(key, 10, 6, 2, 4, std::sync::Arc::from([7_u8; 60]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let layout = test_text_layout_at_scale(
            2.0,
            std::sync::Arc::from([GlyphInstance::new(key, 8, 12)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let mut draws = Vec::new();
        let mut unsupported = GlyphUnsupportedReport::default();

        push_layout_draws(
            &mut draws,
            &mut unsupported,
            crate::scene::SceneOrder::new(1),
            LayoutRect::new(100.0, 50.0, 20.0, 10.0),
            &layout,
            Color::rgb(1, 2, 3),
            &atlas,
        )
        .unwrap();

        assert_eq!(draws.len(), 1);
        assert!(unsupported.is_empty());
        assert_eq!(draws[0].rect, LayoutRect::new(105.0, 54.0, 5.0, 3.0));
    }

    #[test]
    fn missing_atlas_entries_are_counted_and_skipped_during_draw_collection() {
        let atlas = GlyphAtlas::new(8, 8).unwrap();
        let key = test_glyph_key(1);
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let mut draws = Vec::new();
        let mut unsupported = GlyphUnsupportedReport::default();

        push_layout_draws(
            &mut draws,
            &mut unsupported,
            crate::scene::SceneOrder::new(1),
            LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            &layout,
            Color::rgb(1, 2, 3),
            &atlas,
        )
        .unwrap();

        assert!(draws.is_empty());
        assert_eq!(unsupported.missing_atlas_entries, 1);
        assert_eq!(unsupported.skipped_glyph_instances(), 1);
    }

    #[test]
    fn collect_glyph_draws_reuses_scratch_and_resets_unsupported() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 1, 1, 0, 0, std::sync::Arc::from([7_u8]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let valid = prepared_frame_for_layout(test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0), GlyphInstance::new(key, 8, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        ));
        let missing_key = test_glyph_key(2);
        let missing = prepared_frame_for_layout(test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(missing_key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(missing_key)]),
        ));
        let mut draws = Vec::with_capacity(4);

        let unsupported = collect_glyph_draws(&valid, &atlas, &mut draws).unwrap();
        let capacity = draws.capacity();
        let storage = draws.as_ptr();
        assert_eq!(draws.len(), 2);
        assert!(unsupported.is_empty());

        let unsupported = collect_glyph_draws(&missing, &atlas, &mut draws).unwrap();
        assert_eq!(draws.capacity(), capacity);
        assert_eq!(draws.as_ptr(), storage);
        assert!(draws.is_empty());
        assert_eq!(unsupported.missing_atlas_entries, 1);

        let unsupported = collect_glyph_draws(&valid, &atlas, &mut draws).unwrap();
        assert_eq!(draws.capacity(), capacity);
        assert_eq!(draws.as_ptr(), storage);
        assert_eq!(draws.len(), 2);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn collect_glyph_draws_preserves_order_groups_and_per_position_instances() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let first_key = test_glyph_key(1);
        let second_key = test_glyph_key(2);
        atlas
            .ensure_glyph(first_key, |_| {
                Ok(GlyphBitmap::new(
                    first_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([7_u8]),
                ))
            })
            .unwrap();
        atlas
            .ensure_glyph(second_key, |_| {
                Ok(GlyphBitmap::new(
                    second_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([8_u8]),
                ))
            })
            .unwrap();
        let first_layout = test_text_layout(
            std::sync::Arc::from([
                GlyphInstance::new(first_key, 0, 0),
                GlyphInstance::new(first_key, 8, 0),
            ]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(first_key)]),
        );
        let second_layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(second_key, 4, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(second_key)]),
        );
        let prepared = prepared_frame_with_draw_items(vec![
            text_draw_item(crate::scene::SceneOrder::new(1), first_layout),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(2),
                2,
                LayoutRect::new(0.0, 0.0, 10.0, 10.0),
                DrawItemKind::Rect {
                    color: Color::rgb(1, 2, 3),
                },
            ),
            text_draw_item(crate::scene::SceneOrder::new(3), second_layout),
        ]);
        let mut draws = Vec::new();

        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(unsupported.is_empty());
        assert_eq!(draws.len(), 3);
        assert_eq!(draws[0].order, crate::scene::SceneOrder::new(1));
        assert_eq!(draws[1].order, crate::scene::SceneOrder::new(1));
        assert_eq!(draws[2].order, crate::scene::SceneOrder::new(3));
        assert_ne!(draws[0].rect.x(), draws[1].rect.x());
    }

    #[test]
    fn invalid_atlas_entry_uv_returns_resource_failure() {
        let entry = GlyphAtlasEntry {
            x: u32::MAX,
            y: 0,
            width: 1,
            height: 1,
            left: 0,
            top: 0,
        };

        let error = entry.uv(8, 8).unwrap_err();

        assert_eq!(error.kind(), crate::error::ErrorKind::ResourceFailure);
        assert_eq!(error.message(), "glyph atlas uv x overflow");
    }

    fn test_glyph_key(glyph_id: u16) -> GlyphKey {
        test_glyph_key_at_scale(glyph_id, 1.0)
    }

    fn test_glyph_key_at_scale(glyph_id: u16, scale_factor: f32) -> GlyphKey {
        GlyphKey::new(
            cosmic_text::CacheKey::new(
                fontdb::ID::dummy(),
                glyph_id,
                12.0,
                (0.0, 0.0),
                fontdb::Weight::NORMAL,
                cosmic_text::CacheKeyFlags::empty(),
            )
            .0,
            scale_factor,
        )
    }

    fn test_text_layout(
        glyphs: std::sync::Arc<[GlyphInstance]>,
        demands: std::sync::Arc<[crate::text::GlyphDemand]>,
    ) -> crate::text::TextLayoutRef {
        test_text_layout_at_scale(1.0, glyphs, demands)
    }

    fn test_text_layout_at_scale(
        scale_factor: f32,
        glyphs: std::sync::Arc<[GlyphInstance]>,
        demands: std::sync::Arc<[crate::text::GlyphDemand]>,
    ) -> crate::text::TextLayoutRef {
        let key = crate::text::TextLayoutKey {
            node_id: RetainedNodeId::new(1),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text_hash: 1,
            available_inline_width_bits: None,
            font_size_bits: 12.0_f32.to_bits(),
            max_lines: None,
            text_overflow: crate::style::TextOverflow::Clip,
            font_generation: crate::text::FontGeneration::INITIAL,
            scale_generation: 1,
            scale_factor_bits: scale_factor.to_bits(),
        };
        crate::text::TextLayoutRef::new(crate::text::TextLayoutData::new(
            crate::text::TextLayoutGeneration::new(1),
            key,
            crate::text::TextMetrics {
                width: 0.0,
                min_content_width: 0.0,
                max_content_width: 0.0,
                height: 0.0,
                baseline: 0.0,
                line_count: 1,
            },
            glyphs,
            demands,
        ))
    }

    fn text_draw_item(
        order: crate::scene::SceneOrder,
        layout: crate::text::TextLayoutRef,
    ) -> crate::render::DrawItem {
        crate::render::DrawItem::new(
            order,
            order.raw(),
            LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Text {
                text_generation: crate::scene::SceneInputSignature::default(),
                text_metrics_generation: 1,
                layout,
                color: Color::rgb(1, 2, 3),
            },
        )
    }

    fn prepared_frame_for_layout(layout: crate::text::TextLayoutRef) -> PreparedFrame {
        prepared_frame_with_draw_items(vec![text_draw_item(
            crate::scene::SceneOrder::new(1),
            layout,
        )])
    }

    fn prepared_frame_with_draw_items(draw_items: Vec<crate::render::DrawItem>) -> PreparedFrame {
        let draw_orders = draw_items
            .iter()
            .map(crate::render::DrawItem::order)
            .collect();
        let draw_item_count = draw_items.len();
        PreparedFrame::new(
            crate::render::PreparedFrameGeneration::with_surface(
                crate::scene::SceneGeneration::default(),
                1,
            ),
            crate::render::PreparedFrameContext::for_surface(
                crate::layout::Viewport::default(),
                crate::platform::PhysicalSize::new(800, 600),
                1,
            ),
            crate::render::UploadPlan::default(),
            vec![crate::render::PreparedPass::new(
                crate::render::RenderPass::MainColor,
                draw_orders,
                0,
            )],
            draw_items,
            crate::render::FrameGraphStats {
                surface_generation: Some(1),
                pass_count: 1,
                draw_item_count,
                upload_intent_count: 0,
                layer_count: 0,
                unsupported_fragment_count: 0,
                stale_drop_count: 0,
                duration: std::time::Duration::ZERO,
            },
        )
    }
}
