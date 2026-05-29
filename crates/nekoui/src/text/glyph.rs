use std::sync::Arc;

use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::style::TextOverflow;

use super::font::FontGeneration;
use super::measure::{TextGeneration, TextMetrics};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct TextLayoutGeneration(u64);

impl TextLayoutGeneration {
    #[cfg(all(test, target_os = "windows"))]
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw.max(1))
    }

    pub(crate) fn from_layout_key(key: &TextLayoutKey) -> Self {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        Self(hasher.finish().max(1))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct TextLayoutKey {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) node_generation: NodeGeneration,
    pub(crate) text_generation: TextGeneration,
    pub(crate) style_generation: TextGeneration,
    pub(crate) text_hash: u64,
    pub(crate) available_inline_width_bits: Option<u32>,
    pub(crate) font_size_bits: u32,
    pub(crate) max_lines: Option<usize>,
    pub(crate) text_overflow: TextOverflow,
    pub(crate) font_generation: FontGeneration,
    pub(crate) scale_generation: u64,
    pub(crate) scale_factor_bits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct GlyphKey {
    cache_key: cosmic_text::CacheKey,
    scale_factor_bits: u32,
}

impl GlyphKey {
    pub(crate) fn new(cache_key: cosmic_text::CacheKey, scale_factor: f32) -> Self {
        Self {
            cache_key,
            scale_factor_bits: scale_factor.to_bits(),
        }
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn cache_key(self) -> cosmic_text::CacheKey {
        self.cache_key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct GlyphDemand {
    key: GlyphKey,
}

impl GlyphDemand {
    pub(crate) fn new(key: GlyphKey) -> Self {
        Self { key }
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn key(self) -> GlyphKey {
        self.key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlyphInstance {
    key: GlyphKey,
    /// Coordinates are physical pixels from cosmic_text::LayoutGlyph::physical.
    x: i32,
    y: i32,
}

impl GlyphInstance {
    pub(crate) fn new(key: GlyphKey, x: i32, y: i32) -> Self {
        Self { key, x, y }
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn key(self) -> GlyphKey {
        self.key
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn x(self) -> i32 {
        self.x
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn y(self) -> i32 {
        self.y
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextLayoutData {
    generation: TextLayoutGeneration,
    key: TextLayoutKey,
    scale_factor: f32,
    metrics: TextMetrics,
    glyphs: Arc<[GlyphInstance]>,
    demands: Arc<[GlyphDemand]>,
}

impl TextLayoutData {
    pub(crate) fn new(
        generation: TextLayoutGeneration,
        key: TextLayoutKey,
        metrics: TextMetrics,
        glyphs: Arc<[GlyphInstance]>,
        demands: Arc<[GlyphDemand]>,
    ) -> Self {
        let scale_factor = f32::from_bits(key.scale_factor_bits);
        Self {
            generation,
            key,
            scale_factor,
            metrics,
            glyphs,
            demands,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextLayoutRef {
    data: Arc<TextLayoutData>,
}

impl TextLayoutRef {
    pub(crate) fn new(data: TextLayoutData) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> TextLayoutGeneration {
        self.data.generation
    }

    #[cfg(test)]
    pub(crate) fn key(&self) -> &TextLayoutKey {
        &self.data.key
    }

    pub(crate) fn metrics(&self) -> TextMetrics {
        self.data.metrics
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn scale_factor(&self) -> f32 {
        self.data.scale_factor
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn glyphs(&self) -> &[GlyphInstance] {
        &self.data.glyphs
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn glyph_demands(&self) -> &[GlyphDemand] {
        &self.data.demands
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextGlyphDemand {
    layout: TextLayoutRef,
}

impl TextGlyphDemand {
    pub(crate) fn new(layout: TextLayoutRef) -> Self {
        Self { layout }
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn layout(&self) -> &TextLayoutRef {
        &self.layout
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlyphBitmap {
    key: GlyphKey,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
    pixels: Arc<[u8]>,
}

#[cfg(target_os = "windows")]
impl GlyphBitmap {
    pub(crate) fn new(
        key: GlyphKey,
        width: u32,
        height: u32,
        left: i32,
        top: i32,
        pixels: Arc<[u8]>,
    ) -> Self {
        Self {
            key,
            width,
            height,
            left,
            top,
            pixels,
        }
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn left(&self) -> i32 {
        self.left
    }

    pub(crate) fn top(&self) -> i32 {
        self.top
    }

    pub(crate) fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GlyphRasterError {
    MissingGlyph,
    UnsupportedContent(&'static str),
}
