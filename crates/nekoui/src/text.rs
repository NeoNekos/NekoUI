mod font;
mod glyph;
mod measure;

pub(crate) use font::{FontGeneration, FontManager};
#[cfg(target_os = "windows")]
pub(crate) use glyph::{GlyphBitmap, GlyphRasterError};
pub(crate) use glyph::{
    GlyphDemand, GlyphInstance, GlyphKey, TextGlyphDemand, TextLayoutData, TextLayoutGeneration,
    TextLayoutKey, TextLayoutRef,
};
#[cfg(test)]
pub(crate) use measure::TextMeasureResult;
#[cfg(all(test, target_os = "windows"))]
pub(crate) use measure::TextMetrics;
pub(crate) use measure::{
    TextGeneration, TextLayoutResult, TextMeasureQuery, TextMeasureSession, TextMeasureStats,
};

#[cfg(test)]
mod tests;
