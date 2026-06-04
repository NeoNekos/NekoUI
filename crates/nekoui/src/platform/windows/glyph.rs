use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

use crate::error::{NekoError, NekoResult};
use crate::layout::LayoutRect;
use crate::render::{DrawItemKind, PreparedFrame};
use crate::style::Color;
use crate::text::{
    FontManager, GlyphBitmap, GlyphBitmapFormat, GlyphInstance, GlyphKey, GlyphRasterError,
    TextLayoutRef,
};

use super::clip::{ActiveClip, ClipStack};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlyphDrawFormat {
    MonoMask,
    ColorRgba,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GlyphDrawPlan {
    mono_draws: Vec<GlyphDraw>,
    color_draws: Vec<GlyphDraw>,
    runs: Vec<GlyphDrawRun>,
}

impl GlyphDrawPlan {
    pub(super) fn clear(&mut self) {
        self.mono_draws.clear();
        self.color_draws.clear();
        self.runs.clear();
    }

    pub(super) fn has_mono_draws(&self) -> bool {
        !self.mono_draws.is_empty()
    }

    pub(super) fn has_color_draws(&self) -> bool {
        !self.color_draws.is_empty()
    }

    pub(super) fn runs(&self) -> &[GlyphDrawRun] {
        &self.runs
    }

    pub(super) fn mono_draws(&self) -> &[GlyphDraw] {
        &self.mono_draws
    }

    pub(super) fn color_draws(&self) -> &[GlyphDraw] {
        &self.color_draws
    }

    fn push(&mut self, format: GlyphDrawFormat, draw: GlyphDraw) {
        let index = match format {
            GlyphDrawFormat::MonoMask => {
                let index = self.mono_draws.len();
                self.mono_draws.push(draw);
                index
            }
            GlyphDrawFormat::ColorRgba => {
                let index = self.color_draws.len();
                self.color_draws.push(draw);
                index
            }
        };
        if let Some(run) = self.runs.last_mut()
            && run.order == draw.order
            && run.format == format
            && run.range.end == index
        {
            run.range.end += 1;
            return;
        }
        self.runs.push(GlyphDrawRun {
            order: draw.order,
            format,
            range: index..index + 1,
        });
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct GlyphDrawRun {
    pub(super) order: crate::scene::SceneOrder,
    pub(super) format: GlyphDrawFormat,
    pub(super) range: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GlyphDrawContext {
    order: crate::scene::SceneOrder,
    origin: LayoutRect,
    clip: Option<LayoutRect>,
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
    format: GlyphBitmapFormat,
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
        Self::new_with_format(GlyphBitmapFormat::MaskR8, width, height)
    }

    pub(super) fn new_color_rgba8(width: u32, height: u32) -> NekoResult<Self> {
        Self::new_with_format(GlyphBitmapFormat::ColorRgba8, width, height)
    }

    fn new_with_format(format: GlyphBitmapFormat, width: u32, height: u32) -> NekoResult<Self> {
        let len = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(format.bytes_per_pixel() as u32))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| NekoError::resource_failure("glyph atlas page is too large"))?;
        Ok(Self {
            format,
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

    #[cfg(test)]
    pub(super) fn format(&self) -> GlyphBitmapFormat {
        self.format
    }

    #[cfg(test)]
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
        self.ensure_bitmap(key, bitmap)
    }

    fn ensure_bitmap(
        &mut self,
        key: GlyphKey,
        bitmap: GlyphBitmap,
    ) -> NekoResult<GlyphAtlasOutcome> {
        if let Some(entry) = self.entries.get(&key).copied() {
            return Ok(GlyphAtlasOutcome::Ready(entry));
        }
        if let Some(reason) = self.skipped.get(&key).copied() {
            return Ok(GlyphAtlasOutcome::Unsupported(reason));
        }
        if bitmap.format() != self.format {
            let reason = GlyphSkipReason::UnsupportedContent("glyph_bitmap_format_mismatch");
            self.remember_skip(key, reason);
            return Ok(GlyphAtlasOutcome::Unsupported(reason));
        }
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

    pub(super) fn bytes_per_pixel(&self) -> usize {
        self.format.bytes_per_pixel()
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
            .and_then(|value| value.checked_mul(self.bytes_per_pixel() as u32))
            .and_then(|value| usize::try_from(value).ok())
            .expect("test glyph atlas pixel coordinate is out of bounds");
        self.pixels[offset]
    }

    #[cfg(test)]
    pub(super) fn pixel_bytes(&self, x: u32, y: u32) -> &[u8] {
        let start = y
            .checked_mul(self.width)
            .and_then(|value| value.checked_add(x))
            .and_then(|value| value.checked_mul(self.bytes_per_pixel() as u32))
            .and_then(|value| usize::try_from(value).ok())
            .expect("test glyph atlas pixel coordinate is out of bounds");
        let end = start + self.bytes_per_pixel();
        &self.pixels[start..end]
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
        let row_bytes = checked_usize_mul(
            width,
            bitmap.format().bytes_per_pixel(),
            "glyph bitmap row size overflows usize",
        )?;
        let expected_len =
            checked_usize_mul(row_bytes, height, "glyph bitmap byte size overflows usize")?;
        if bitmap.pixels().len() != expected_len {
            return Err(NekoError::resource_failure(
                "glyph bitmap byte size does not match placement",
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
        let atlas_row_bytes = checked_usize_mul(
            atlas_width,
            self.bytes_per_pixel(),
            "glyph atlas row size overflows usize",
        )?;
        let entry_x = usize::try_from(entry.x)
            .map_err(|_| NekoError::resource_failure("glyph atlas entry x exceeds usize"))?;
        let entry_x_bytes = checked_usize_mul(
            entry_x,
            self.bytes_per_pixel(),
            "glyph atlas entry byte x overflows usize",
        )?;
        let entry_y = usize::try_from(entry.y)
            .map_err(|_| NekoError::resource_failure("glyph atlas entry y exceeds usize"))?;
        for row in 0..height {
            let dst_y = checked_usize_add(entry_y, row, "glyph bitmap destination y overflow")?;
            let dst_row_start = checked_usize_mul(
                dst_y,
                atlas_row_bytes,
                "glyph bitmap destination row overflow",
            )?;
            let dst_start = checked_usize_add(
                dst_row_start,
                entry_x_bytes,
                "glyph bitmap destination start overflow",
            )?;
            let dst_end = checked_usize_add(
                dst_start,
                row_bytes,
                "glyph bitmap destination end overflow",
            )?;
            let src_start = checked_usize_mul(row, row_bytes, "glyph bitmap source row overflow")?;
            let src_end =
                checked_usize_add(src_start, row_bytes, "glyph bitmap source end overflow")?;
            let source = bitmap.pixels().get(src_start..src_end).ok_or_else(|| {
                NekoError::resource_failure("glyph bitmap byte size does not match placement")
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
            bitmap.format().empty_content_kind(),
        )));
    }
    let width = usize::try_from(bitmap.width())
        .map_err(|_| NekoError::resource_failure("glyph bitmap width exceeds usize"))?;
    let height = usize::try_from(bitmap.height())
        .map_err(|_| NekoError::resource_failure("glyph bitmap height exceeds usize"))?;
    let pixel_count = checked_usize_mul(width, height, "glyph bitmap pixel count overflows usize")?;
    let expected_len = checked_usize_mul(
        pixel_count,
        bitmap.format().bytes_per_pixel(),
        "glyph bitmap byte size overflows usize",
    )?;
    if bitmap.pixels().len() != expected_len {
        return Ok(Some(GlyphSkipReason::UnsupportedContent(
            bitmap.format().malformed_content_kind(),
        )));
    }
    Ok(None)
}

enum GlyphAllocation {
    Ready(GlyphAtlasEntry),
    Unsupported(GlyphSkipReason),
}

#[cfg(test)]
pub(super) fn prepare_glyph_atlas(
    prepared: &PreparedFrame,
    atlas: &mut GlyphAtlas,
    font_manager: &FontManager,
) -> NekoResult<GlyphUnsupportedReport> {
    let mut report = GlyphUnsupportedReport::default();
    let visible_text_orders = visible_text_draw_orders(prepared);
    for intent in prepared.upload_plan().intents() {
        let Some(glyphs) = intent.glyphs() else {
            continue;
        };
        if !intent
            .dependent_draw_orders()
            .iter()
            .any(|order| visible_text_orders.contains(order))
        {
            continue;
        }
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

pub(super) fn prepare_glyph_atlases(
    prepared: &PreparedFrame,
    mono_atlas: &mut GlyphAtlas,
    color_atlas: &mut Option<GlyphAtlas>,
    font_manager: &FontManager,
) -> NekoResult<GlyphUnsupportedReport> {
    let mut report = GlyphUnsupportedReport::default();
    let visible_text_orders = visible_text_draw_orders(prepared);
    for intent in prepared.upload_plan().intents() {
        let Some(glyphs) = intent.glyphs() else {
            continue;
        };
        if !intent
            .dependent_draw_orders()
            .iter()
            .any(|order| visible_text_orders.contains(order))
        {
            continue;
        }
        for demand in glyphs.layout().glyph_demands() {
            let key = demand.key();
            if mono_atlas.entries.contains_key(&key)
                || color_atlas
                    .as_ref()
                    .is_some_and(|atlas| atlas.entries.contains_key(&key))
            {
                continue;
            }
            if let Some(reason) = mono_atlas
                .skipped
                .get(&key)
                .or_else(|| {
                    color_atlas
                        .as_ref()
                        .and_then(|atlas| atlas.skipped.get(&key))
                })
                .copied()
            {
                report.note_skip(reason);
                continue;
            }
            let bitmap = match font_manager.rasterize_glyph(key) {
                Ok(bitmap) => bitmap,
                Err(GlyphRasterError::MissingGlyph) => {
                    let reason = GlyphSkipReason::MissingGlyph;
                    mono_atlas.remember_skip(key, reason);
                    if let Some(color_atlas) = color_atlas.as_mut() {
                        color_atlas.remember_skip(key, reason);
                    }
                    report.note_skip(reason);
                    continue;
                }
                Err(GlyphRasterError::UnsupportedContent(kind)) => {
                    let reason = GlyphSkipReason::UnsupportedContent(kind);
                    mono_atlas.remember_skip(key, reason);
                    if let Some(color_atlas) = color_atlas.as_mut() {
                        color_atlas.remember_skip(key, reason);
                    }
                    report.note_skip(reason);
                    continue;
                }
            };
            let target = match bitmap.format() {
                GlyphBitmapFormat::MaskR8 => &mut *mono_atlas,
                GlyphBitmapFormat::ColorRgba8 => {
                    if color_atlas.is_none() {
                        *color_atlas = Some(GlyphAtlas::new_color_rgba8(
                            GLYPH_ATLAS_WIDTH,
                            GLYPH_ATLAS_HEIGHT,
                        )?);
                    }
                    color_atlas.as_mut().ok_or_else(|| {
                        NekoError::resource_failure("color glyph atlas was not materialized")
                    })?
                }
            };
            if let GlyphAtlasOutcome::Unsupported(reason) = target.ensure_bitmap(key, bitmap)? {
                report.note_skip(reason);
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
pub(super) fn collect_glyph_draws(
    prepared: &PreparedFrame,
    atlas: &GlyphAtlas,
    draws: &mut Vec<GlyphDraw>,
) -> NekoResult<GlyphUnsupportedReport> {
    draws.clear();
    let mut unsupported = GlyphUnsupportedReport::default();
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            DrawItemKind::Text {
                layout,
                clip,
                color,
                ..
            } => {
                if color.to_current_backend_sdr_srgb_rgba().is_none() {
                    continue;
                }
                let Some(clip) = glyph_clip(*clip, clip_stack.active_clip()) else {
                    continue;
                };
                push_layout_draws(
                    draws,
                    &mut unsupported,
                    GlyphDrawContext {
                        order: item.order(),
                        origin: item.rect(),
                        clip,
                    },
                    layout,
                    *color,
                    atlas,
                )?;
            }
            DrawItemKind::BoxShape { .. }
            | DrawItemKind::Rect { .. }
            | DrawItemKind::Unsupported { .. } => {}
        }
    }
    Ok(unsupported)
}

pub(super) fn collect_glyph_draw_plan(
    prepared: &PreparedFrame,
    mono_atlas: &GlyphAtlas,
    color_atlas: Option<&GlyphAtlas>,
    plan: &mut GlyphDrawPlan,
) -> NekoResult<GlyphUnsupportedReport> {
    plan.clear();
    let mut unsupported = GlyphUnsupportedReport::default();
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            DrawItemKind::Text {
                layout,
                clip,
                color,
                ..
            } => {
                if color.to_current_backend_sdr_srgb_rgba().is_none() {
                    continue;
                }
                let Some(clip) = glyph_clip(*clip, clip_stack.active_clip()) else {
                    continue;
                };
                push_layout_draw_plan(
                    plan,
                    &mut unsupported,
                    GlyphDrawContext {
                        order: item.order(),
                        origin: item.rect(),
                        clip,
                    },
                    layout,
                    *color,
                    mono_atlas,
                    color_atlas,
                )?;
            }
            DrawItemKind::BoxShape { .. }
            | DrawItemKind::Rect { .. }
            | DrawItemKind::Unsupported { .. } => {}
        }
    }
    Ok(unsupported)
}

fn visible_text_draw_orders(prepared: &PreparedFrame) -> BTreeSet<crate::scene::SceneOrder> {
    let mut visible = BTreeSet::new();
    let mut clip_stack = ClipStack::default();
    for item in prepared.draw_items() {
        match item.kind() {
            DrawItemKind::ClipPush { clip } => clip_stack.push(*clip),
            DrawItemKind::ClipPop => clip_stack.pop(),
            DrawItemKind::Text { color, clip, .. }
                if color.to_current_backend_sdr_srgb_rgba().is_some()
                    && glyph_clip(*clip, clip_stack.active_clip()).is_some() =>
            {
                visible.insert(item.order());
            }
            DrawItemKind::BoxShape { .. }
            | DrawItemKind::Rect { .. }
            | DrawItemKind::Text { .. }
            | DrawItemKind::Unsupported { .. } => {}
        }
    }
    visible
}

fn glyph_clip(text_clip: Option<LayoutRect>, scene_clip: ActiveClip) -> Option<Option<LayoutRect>> {
    match (text_clip, scene_clip) {
        (_, ActiveClip::Empty) => None,
        (None, ActiveClip::Unclipped) => Some(None),
        (Some(clip), ActiveClip::Unclipped) | (None, ActiveClip::Rect(clip)) => Some(Some(clip)),
        (Some(text_clip), ActiveClip::Rect(scene_clip)) => {
            text_clip.intersect(scene_clip).map(Some)
        }
    }
}

#[cfg(test)]
fn push_layout_draws(
    draws: &mut Vec<GlyphDraw>,
    unsupported: &mut GlyphUnsupportedReport,
    context: GlyphDrawContext,
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
        let rect = glyph_rect(context.origin, *glyph, entry, scale_factor);
        let uv = entry.uv(atlas.width, atlas.height)?;
        let Some((rect, uv)) = clip_glyph_draw(rect, uv, context.clip) else {
            continue;
        };
        draws.push(GlyphDraw {
            order: context.order,
            rect,
            uv,
            color,
        });
    }
    Ok(())
}

fn push_layout_draw_plan(
    plan: &mut GlyphDrawPlan,
    unsupported: &mut GlyphUnsupportedReport,
    context: GlyphDrawContext,
    layout: &TextLayoutRef,
    color: Color,
    mono_atlas: &GlyphAtlas,
    color_atlas: Option<&GlyphAtlas>,
) -> NekoResult<()> {
    let scale_factor = layout.scale_factor();
    for glyph in layout.glyphs() {
        let Some((format, entry, atlas_width, atlas_height)) =
            glyph_entry(glyph.key(), mono_atlas, color_atlas, unsupported)
        else {
            continue;
        };
        if entry.width == 0 || entry.height == 0 {
            continue;
        }
        let rect = glyph_rect(context.origin, *glyph, entry, scale_factor);
        let uv = entry.uv(atlas_width, atlas_height)?;
        let Some((rect, uv)) = clip_glyph_draw(rect, uv, context.clip) else {
            continue;
        };
        plan.push(
            format,
            GlyphDraw {
                order: context.order,
                rect,
                uv,
                color,
            },
        );
    }
    Ok(())
}

fn glyph_entry(
    key: GlyphKey,
    mono_atlas: &GlyphAtlas,
    color_atlas: Option<&GlyphAtlas>,
    unsupported: &mut GlyphUnsupportedReport,
) -> Option<(GlyphDrawFormat, GlyphAtlasEntry, u32, u32)> {
    if let Some(entry) = mono_atlas.entries.get(&key).copied() {
        return Some((
            GlyphDrawFormat::MonoMask,
            entry,
            mono_atlas.width,
            mono_atlas.height,
        ));
    }
    if let Some(color_atlas) = color_atlas
        && let Some(entry) = color_atlas.entries.get(&key).copied()
    {
        return Some((
            GlyphDrawFormat::ColorRgba,
            entry,
            color_atlas.width,
            color_atlas.height,
        ));
    }
    if let Some(reason) = mono_atlas
        .skipped
        .get(&key)
        .or_else(|| color_atlas.and_then(|atlas| atlas.skipped.get(&key)))
        .copied()
    {
        unsupported.note_skip(reason);
        unsupported.skipped_cached_glyph_instances += 1;
    } else {
        unsupported.missing_atlas_entries += 1;
    }
    None
}

fn clip_glyph_draw(
    rect: LayoutRect,
    uv: GlyphUv,
    clip: Option<LayoutRect>,
) -> Option<(LayoutRect, GlyphUv)> {
    let Some(clip) = clip else {
        return Some((rect, uv));
    };
    let clipped = rect.intersect(clip)?;
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    let u_span = uv.right - uv.left;
    let v_span = uv.bottom - uv.top;
    let left_ratio = (clipped.x() - rect.x()) / rect.width();
    let right_ratio = (clipped.x() + clipped.width() - rect.x()) / rect.width();
    let top_ratio = (clipped.y() - rect.y()) / rect.height();
    let bottom_ratio = (clipped.y() + clipped.height() - rect.y()) / rect.height();
    Some((
        clipped,
        GlyphUv {
            left: uv.left + u_span * left_ratio,
            top: uv.top + v_span * top_ratio,
            right: uv.left + u_span * right_ratio,
            bottom: uv.top + v_span * bottom_ratio,
        },
    ))
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
    use crate::text::{
        TextGeneration, TextInlineConstraint, TextLayoutResult, TextMeasureQuery,
        TextMeasureSession,
    };
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
            inline_constraint: TextInlineConstraint::MaxContent,
            layout_mode: crate::text::TextLayoutMode::SoftWrap,
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
        assert_eq!(bitmap.format(), GlyphBitmapFormat::MaskR8);
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
                Err(GlyphRasterError::UnsupportedContent("subpixel_mask"))
            })
            .unwrap();

        assert_eq!(
            missing,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::MissingGlyph)
        );
        assert_eq!(
            unsupported,
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent("subpixel_mask"))
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
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent("subpixel_mask"))
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
    fn color_glyph_bitmaps_are_accepted_as_rgba8_and_preserve_channel_order() {
        let mut atlas = GlyphAtlas::new_color_rgba8(8, 8).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new_color_rgba8(
            key,
            2,
            1,
            0,
            0,
            std::sync::Arc::from([10_u8, 20, 30, 40, 50, 60, 70, 80]),
        );

        let entry = match atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };

        assert_eq!(atlas.format(), GlyphBitmapFormat::ColorRgba8);
        assert_eq!(entry.width, 2);
        assert_eq!(entry.height, 1);
        assert_eq!(atlas.pixel_bytes(entry.x, entry.y), &[10, 20, 30, 40]);
        assert_eq!(atlas.pixel_bytes(entry.x + 1, entry.y), &[50, 60, 70, 80]);
    }

    #[test]
    fn malformed_color_glyph_bitmaps_are_rejected_without_poisoning_mono_atlas() {
        let mut mono_atlas = GlyphAtlas::new(8, 8).unwrap();
        let mut color_atlas = GlyphAtlas::new_color_rgba8(8, 8).unwrap();
        let mono_key = test_glyph_key(1);
        let color_key = test_glyph_key(2);
        let mono = GlyphBitmap::new(mono_key, 1, 1, 0, 0, std::sync::Arc::from([7_u8]));
        let malformed_color =
            GlyphBitmap::new_color_rgba8(color_key, 2, 2, 0, 0, std::sync::Arc::from([1_u8, 2, 3]));

        assert!(matches!(
            mono_atlas.ensure_glyph(mono_key, |_| Ok(mono)).unwrap(),
            GlyphAtlasOutcome::Ready(_)
        ));
        assert_eq!(
            color_atlas
                .ensure_glyph(color_key, |_| Ok(malformed_color))
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "malformed_color_glyph_bitmap"
            ))
        );

        assert!(mono_atlas.entries.contains_key(&mono_key));
        assert!(!mono_atlas.skipped.contains_key(&color_key));
        assert!(color_atlas.pixels().iter().all(|pixel| *pixel == 0));
    }

    #[test]
    fn color_atlas_empty_oversize_and_full_paths_are_nonfatal() {
        let mut atlas = GlyphAtlas::new_color_rgba8(8, 8).unwrap();
        let first_key = test_glyph_key(1);
        let empty_key = test_glyph_key(2);
        let oversize_key = test_glyph_key(3);
        let first = GlyphBitmap::new_color_rgba8(
            first_key,
            2,
            2,
            0,
            0,
            std::sync::Arc::from([5_u8; 2 * 2 * 4]),
        );
        let first_entry = match atlas.ensure_glyph(first_key, |_| Ok(first)).unwrap() {
            GlyphAtlasOutcome::Ready(entry) => entry,
            GlyphAtlasOutcome::Unsupported(_) => unreachable!(),
        };
        assert_eq!(
            atlas
                .ensure_glyph(empty_key, |_| {
                    Ok(GlyphBitmap::new_color_rgba8(
                        empty_key,
                        0,
                        0,
                        0,
                        0,
                        std::sync::Arc::from([]),
                    ))
                })
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::UnsupportedContent(
                "empty_color_glyph_bitmap"
            ))
        );
        assert_eq!(
            atlas
                .ensure_glyph(oversize_key, |_| {
                    Ok(GlyphBitmap::new_color_rgba8(
                        oversize_key,
                        7,
                        1,
                        0,
                        0,
                        std::sync::Arc::from([9_u8; 7 * 4]),
                    ))
                })
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::ExceedsAtlasPage)
        );
        assert_eq!(
            atlas.ensure_glyph(first_key, |_| unreachable!()).unwrap(),
            GlyphAtlasOutcome::Ready(first_entry)
        );

        let mut full_atlas = GlyphAtlas::new_color_rgba8(4, 4).unwrap();
        let full_first_key = test_glyph_key(4);
        let full_second_key = test_glyph_key(5);
        assert!(matches!(
            full_atlas
                .ensure_glyph(full_first_key, |_| {
                    Ok(GlyphBitmap::new_color_rgba8(
                        full_first_key,
                        1,
                        1,
                        0,
                        0,
                        std::sync::Arc::from([1_u8, 2, 3, 4]),
                    ))
                })
                .unwrap(),
            GlyphAtlasOutcome::Ready(_)
        ));
        assert_eq!(
            full_atlas
                .ensure_glyph(full_second_key, |_| {
                    Ok(GlyphBitmap::new_color_rgba8(
                        full_second_key,
                        1,
                        1,
                        0,
                        0,
                        std::sync::Arc::from([5_u8, 6, 7, 8]),
                    ))
                })
                .unwrap(),
            GlyphAtlasOutcome::Unsupported(GlyphSkipReason::AtlasFull)
        );
        assert!(!full_atlas.skipped.contains_key(&full_second_key));
    }

    #[test]
    fn mono_and_color_atlas_paths_stay_distinct() {
        let mut mono_atlas = GlyphAtlas::new(8, 8).unwrap();
        let mut color_atlas = GlyphAtlas::new_color_rgba8(8, 8).unwrap();
        let mono_key = test_glyph_key(1);
        let color_key = test_glyph_key(2);

        mono_atlas
            .ensure_glyph(mono_key, |_| {
                Ok(GlyphBitmap::new(
                    mono_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([11_u8]),
                ))
            })
            .unwrap();
        color_atlas
            .ensure_glyph(color_key, |_| {
                Ok(GlyphBitmap::new_color_rgba8(
                    color_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([1_u8, 2, 3, 4]),
                ))
            })
            .unwrap();

        assert!(mono_atlas.entries.contains_key(&mono_key));
        assert!(!mono_atlas.entries.contains_key(&color_key));
        assert!(color_atlas.entries.contains_key(&color_key));
        assert!(!color_atlas.entries.contains_key(&mono_key));
        assert_eq!(mono_atlas.bytes_per_pixel(), 1);
        assert_eq!(color_atlas.bytes_per_pixel(), 4);
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
            GlyphDrawContext {
                order: crate::scene::SceneOrder::new(1),
                origin: LayoutRect::new(100.0, 50.0, 20.0, 10.0),
                clip: None,
            },
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
    fn collect_glyph_draws_accepts_oklch_text_color() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 8, 4, 0, 0, std::sync::Arc::from([7_u8; 32]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let prepared = prepared_frame_with_draw_items(vec![crate::render::DrawItem::new(
            crate::scene::SceneOrder::new(1),
            1,
            LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Text {
                text_generation: crate::scene::SceneInputSignature::default(),
                text_metrics_generation: 1,
                layout,
                clip: None,
                color: Color::oklch(0.5, 0.1, 120.0),
            },
        )]);
        let mut draws = Vec::new();

        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(unsupported.is_empty());
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].color, Color::oklch(0.5, 0.1, 120.0));
    }

    #[test]
    fn clipped_glyph_draws_adjust_rect_and_uv_to_text_clip() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 8, 4, 0, 0, std::sync::Arc::from([7_u8; 32]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let mut draws = Vec::new();
        let mut unsupported = GlyphUnsupportedReport::default();

        push_layout_draws(
            &mut draws,
            &mut unsupported,
            GlyphDrawContext {
                order: crate::scene::SceneOrder::new(1),
                origin: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
                clip: Some(LayoutRect::new(2.0, 1.0, 4.0, 2.0)),
            },
            &layout,
            Color::rgb(1, 2, 3),
            &atlas,
        )
        .unwrap();

        assert_eq!(draws.len(), 1);
        assert!(unsupported.is_empty());
        assert_eq!(draws[0].rect, LayoutRect::new(2.0, 1.0, 4.0, 2.0));
        assert_eq!(draws[0].uv.left, 0.25);
        assert_eq!(draws[0].uv.top, 0.1875);
        assert_eq!(draws[0].uv.right, 0.5);
        assert_eq!(draws[0].uv.bottom, 0.3125);
    }

    #[test]
    fn glyph_draws_use_intersection_of_scene_clip_and_text_clip() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 8, 4, 0, 0, std::sync::Arc::from([7_u8; 32]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let prepared = prepared_frame_with_draw_items(vec![
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(1),
                1,
                LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                DrawItemKind::ClipPush {
                    clip: LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                },
            ),
            text_draw_item_with_clip(
                crate::scene::SceneOrder::new(2),
                layout,
                Some(LayoutRect::new(2.0, 1.0, 5.0, 3.0)),
            ),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(3),
                1,
                LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                DrawItemKind::ClipPop,
            ),
        ]);
        let mut draws = Vec::new();

        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(unsupported.is_empty());
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].rect, LayoutRect::new(2.0, 2.0, 2.0, 2.0));
        assert_eq!(draws[0].uv.left, 0.25);
        assert_eq!(draws[0].uv.top, 0.25);
        assert_eq!(draws[0].uv.right, 0.375);
        assert_eq!(draws[0].uv.bottom, 0.375);
    }

    #[test]
    fn fully_clipped_glyph_draws_are_skipped_without_unsupported_diagnostics() {
        let mut atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let bitmap = GlyphBitmap::new(key, 2, 2, 0, 0, std::sync::Arc::from([7_u8; 4]));
        atlas.ensure_glyph(key, |_| Ok(bitmap)).unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let mut draws = Vec::new();
        let mut unsupported = GlyphUnsupportedReport::default();

        push_layout_draws(
            &mut draws,
            &mut unsupported,
            GlyphDrawContext {
                order: crate::scene::SceneOrder::new(1),
                origin: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
                clip: Some(LayoutRect::new(5.0, 5.0, 2.0, 2.0)),
            },
            &layout,
            Color::rgb(1, 2, 3),
            &atlas,
        )
        .unwrap();

        assert!(draws.is_empty());
        assert!(unsupported.is_empty());
    }

    #[test]
    fn fully_scene_clipped_glyph_draws_are_skipped_without_atlas_diagnostics() {
        let atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let prepared = prepared_frame_with_draw_items(vec![
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(1),
                1,
                LayoutRect::new(0.0, 0.0, 1.0, 1.0),
                DrawItemKind::ClipPush {
                    clip: LayoutRect::new(0.0, 0.0, 1.0, 1.0),
                },
            ),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(2),
                1,
                LayoutRect::new(10.0, 10.0, 1.0, 1.0),
                DrawItemKind::ClipPush {
                    clip: LayoutRect::new(10.0, 10.0, 1.0, 1.0),
                },
            ),
            text_draw_item(crate::scene::SceneOrder::new(3), layout),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(4),
                1,
                LayoutRect::new(10.0, 10.0, 1.0, 1.0),
                DrawItemKind::ClipPop,
            ),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(5),
                1,
                LayoutRect::new(0.0, 0.0, 1.0, 1.0),
                DrawItemKind::ClipPop,
            ),
        ]);
        let mut draws = Vec::new();

        let unsupported = collect_glyph_draws(&prepared, &atlas, &mut draws).unwrap();

        assert!(draws.is_empty());
        assert!(unsupported.is_empty());
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
            GlyphDrawContext {
                order: crate::scene::SceneOrder::new(1),
                origin: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
                clip: None,
            },
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
    fn collect_glyph_draw_plan_preserves_mixed_color_and_mono_glyph_order() {
        let mut mono_atlas = GlyphAtlas::new(16, 16).unwrap();
        let mut color_atlas = GlyphAtlas::new_color_rgba8(16, 16).unwrap();
        let mono_key = test_glyph_key(1);
        let color_key = test_glyph_key(2);
        mono_atlas
            .ensure_glyph(mono_key, |_| {
                Ok(GlyphBitmap::new(
                    mono_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([7_u8]),
                ))
            })
            .unwrap();
        color_atlas
            .ensure_glyph(color_key, |_| {
                Ok(GlyphBitmap::new_color_rgba8(
                    color_key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([1_u8, 2, 3, 4]),
                ))
            })
            .unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([
                GlyphInstance::new(mono_key, 0, 0),
                GlyphInstance::new(color_key, 8, 0),
                GlyphInstance::new(mono_key, 16, 0),
            ]),
            std::sync::Arc::from([
                crate::text::GlyphDemand::new(mono_key),
                crate::text::GlyphDemand::new(color_key),
            ]),
        );
        let prepared = prepared_frame_for_layout(layout);
        let mut plan = GlyphDrawPlan::default();

        let unsupported =
            collect_glyph_draw_plan(&prepared, &mono_atlas, Some(&color_atlas), &mut plan).unwrap();

        assert!(unsupported.is_empty());
        assert_eq!(plan.mono_draws().len(), 2);
        assert_eq!(plan.color_draws().len(), 1);
        assert!(plan.has_mono_draws());
        assert!(plan.has_color_draws());
        assert_eq!(plan.runs().len(), 3);
        assert_eq!(plan.runs()[0].format, GlyphDrawFormat::MonoMask);
        assert_eq!(plan.runs()[0].range, 0..1);
        assert_eq!(plan.runs()[1].format, GlyphDrawFormat::ColorRgba);
        assert_eq!(plan.runs()[1].range, 0..1);
        assert_eq!(plan.runs()[2].format, GlyphDrawFormat::MonoMask);
        assert_eq!(plan.runs()[2].range, 1..2);
        assert!(plan.mono_draws()[0].rect.x() < plan.color_draws()[0].rect.x());
        assert!(plan.color_draws()[0].rect.x() < plan.mono_draws()[1].rect.x());
    }

    #[test]
    fn mono_only_glyph_draw_plan_does_not_require_color_atlas() {
        let mut mono_atlas = GlyphAtlas::new(16, 16).unwrap();
        let key = test_glyph_key(1);
        mono_atlas
            .ensure_glyph(key, |_| {
                Ok(GlyphBitmap::new(
                    key,
                    1,
                    1,
                    0,
                    0,
                    std::sync::Arc::from([7_u8]),
                ))
            })
            .unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let prepared = prepared_frame_for_layout(layout);
        let mut plan = GlyphDrawPlan::default();

        let unsupported = collect_glyph_draw_plan(&prepared, &mono_atlas, None, &mut plan).unwrap();

        assert!(unsupported.is_empty());
        assert!(plan.has_mono_draws());
        assert!(!plan.has_color_draws());
        assert_eq!(plan.runs().len(), 1);
        assert_eq!(plan.runs()[0].format, GlyphDrawFormat::MonoMask);
    }

    #[test]
    fn color_glyph_draw_plan_respects_scene_and_text_clip_collection() {
        let mono_atlas = GlyphAtlas::new(16, 16).unwrap();
        let mut color_atlas = GlyphAtlas::new_color_rgba8(16, 16).unwrap();
        let key = test_glyph_key(1);
        color_atlas
            .ensure_glyph(key, |_| {
                Ok(GlyphBitmap::new_color_rgba8(
                    key,
                    8,
                    4,
                    0,
                    0,
                    std::sync::Arc::from([9_u8; 8 * 4 * 4]),
                ))
            })
            .unwrap();
        let layout = test_text_layout(
            std::sync::Arc::from([GlyphInstance::new(key, 0, 0)]),
            std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
        );
        let prepared = prepared_frame_with_draw_items(vec![
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(1),
                1,
                LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                DrawItemKind::ClipPush {
                    clip: LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                },
            ),
            text_draw_item_with_clip(
                crate::scene::SceneOrder::new(2),
                layout,
                Some(LayoutRect::new(2.0, 1.0, 5.0, 3.0)),
            ),
            crate::render::DrawItem::new(
                crate::scene::SceneOrder::new(3),
                1,
                LayoutRect::new(0.0, 2.0, 4.0, 2.0),
                DrawItemKind::ClipPop,
            ),
        ]);
        let mut plan = GlyphDrawPlan::default();

        let unsupported =
            collect_glyph_draw_plan(&prepared, &mono_atlas, Some(&color_atlas), &mut plan).unwrap();

        assert!(unsupported.is_empty());
        assert!(plan.mono_draws().is_empty());
        assert_eq!(plan.color_draws().len(), 1);
        assert_eq!(plan.runs().len(), 1);
        assert_eq!(plan.runs()[0].format, GlyphDrawFormat::ColorRgba);
        assert_eq!(
            plan.color_draws()[0].rect,
            LayoutRect::new(2.0, 2.0, 2.0, 2.0)
        );
        assert_eq!(plan.color_draws()[0].uv.left, 0.25);
        assert_eq!(plan.color_draws()[0].uv.top, 0.25);
        assert_eq!(plan.color_draws()[0].uv.right, 0.375);
        assert_eq!(plan.color_draws()[0].uv.bottom, 0.375);
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
            inline_constraint: TextInlineConstraint::MaxContent.cache_key(),
            layout_mode: crate::text::TextLayoutMode::SoftWrap,
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
            LayoutRect::new(0.0, 0.0, 1.0, 1.0),
            glyphs,
            demands,
        ))
    }

    fn text_draw_item(
        order: crate::scene::SceneOrder,
        layout: crate::text::TextLayoutRef,
    ) -> crate::render::DrawItem {
        text_draw_item_with_clip(order, layout, None)
    }

    fn text_draw_item_with_clip(
        order: crate::scene::SceneOrder,
        layout: crate::text::TextLayoutRef,
        clip: Option<LayoutRect>,
    ) -> crate::render::DrawItem {
        crate::render::DrawItem::new(
            order,
            order.raw(),
            LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Text {
                text_generation: crate::scene::SceneInputSignature::default(),
                text_metrics_generation: 1,
                layout,
                clip,
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
                box_shape_count: 0,
                unsupported_fragment_count: 0,
                stale_drop_count: 0,
                duration: std::time::Duration::ZERO,
            },
        )
    }
}
