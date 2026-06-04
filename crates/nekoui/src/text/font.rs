use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct FontGeneration(u64);

impl FontGeneration {
    pub(crate) const INITIAL: Self = Self(1);

    #[cfg(test)]
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct FontBlobId(u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FontMetadata {
    id: FontBlobId,
    family: &'static str,
}

impl FontMetadata {
    fn deterministic_fallback() -> Self {
        Self {
            id: FontBlobId(1),
            family: "NekoUI deterministic fallback",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FontBlobRef {
    id: FontBlobId,
    bytes: Arc<[u8]>,
    face_index: u32,
}

impl FontBlobRef {
    #[cfg(test)]
    pub(crate) fn new_for_test(id: u64, bytes: Arc<[u8]>, face_index: u32) -> Self {
        Self {
            id: FontBlobId(id),
            bytes,
            face_index,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FontFallbackSnapshot {
    generation: FontGeneration,
    entries: Arc<[FontMetadata]>,
    blobs: Arc<[FontBlobRef]>,
}

impl FontFallbackSnapshot {
    #[cfg(test)]
    pub(crate) fn generation(&self) -> FontGeneration {
        self.generation
    }

    pub(crate) fn metadata_count(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn blob_count(&self) -> usize {
        self.blobs.len()
    }
}

pub(crate) struct FontManager {
    generation: FontGeneration,
    fallback_snapshot: FontFallbackSnapshot,
    _database: fontdb::Database,
    font_system: RefCell<cosmic_text::FontSystem>,
    #[cfg(target_os = "windows")]
    swash_cache: RefCell<cosmic_text::SwashCache>,
}

impl fmt::Debug for FontManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FontManager")
            .field("generation", &self.generation)
            .field("fallback_snapshot", &self.fallback_snapshot)
            .finish_non_exhaustive()
    }
}

impl Default for FontManager {
    fn default() -> Self {
        let generation = FontGeneration::INITIAL;
        Self {
            generation,
            fallback_snapshot: FontFallbackSnapshot {
                generation,
                entries: Arc::from([FontMetadata::deterministic_fallback()]),
                blobs: Arc::from([]),
            },
            _database: fontdb::Database::new(),
            font_system: RefCell::new(cosmic_text::FontSystem::new()),
            #[cfg(target_os = "windows")]
            swash_cache: RefCell::new(cosmic_text::SwashCache::new()),
        }
    }
}

impl FontManager {
    pub(crate) fn generation(&self) -> FontGeneration {
        self.generation
    }

    pub(crate) fn fallback_snapshot(&self) -> FontFallbackSnapshot {
        self.fallback_snapshot.clone()
    }

    pub(crate) fn with_font_system<R>(
        &self,
        f: impl FnOnce(&mut cosmic_text::FontSystem) -> R,
    ) -> R {
        f(&mut self.font_system.borrow_mut())
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn rasterize_glyph(
        &self,
        key: crate::text::GlyphKey,
    ) -> Result<crate::text::GlyphBitmap, crate::text::GlyphRasterError> {
        use cosmic_text::SwashContent;

        let mut font_system = self.font_system.borrow_mut();
        let mut swash_cache = self.swash_cache.borrow_mut();
        let Some(image) = swash_cache
            .get_image(&mut font_system, key.cache_key())
            .as_ref()
        else {
            return Err(crate::text::GlyphRasterError::MissingGlyph);
        };
        match image.content {
            SwashContent::Mask => Ok(crate::text::GlyphBitmap::new(
                key,
                image.placement.width,
                image.placement.height,
                image.placement.left,
                image.placement.top,
                Arc::from(image.data.as_slice()),
            )),
            SwashContent::Color => Ok(crate::text::GlyphBitmap::new_color_rgba8(
                key,
                image.placement.width,
                image.placement.height,
                image.placement.left,
                image.placement.top,
                Arc::from(image.data.as_slice()),
            )),
            SwashContent::SubpixelMask => Err(crate::text::GlyphRasterError::UnsupportedContent(
                "subpixel_mask",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn bump_generation_for_test(&mut self) {
        self.generation = self.generation.next();
        self.fallback_snapshot.generation = self.generation;
    }

    #[cfg(test)]
    pub(crate) fn install_test_blob(&mut self, blob: FontBlobRef) {
        self.generation = self.generation.next();
        self.fallback_snapshot = FontFallbackSnapshot {
            generation: self.generation,
            entries: self.fallback_snapshot.entries.clone(),
            blobs: Arc::from([blob]),
        };
    }

    #[cfg(test)]
    pub(crate) fn clear_fallback_for_test(&mut self) {
        self.generation = self.generation.next();
        self.fallback_snapshot = FontFallbackSnapshot {
            generation: self.generation,
            entries: Arc::from([]),
            blobs: Arc::from([]),
        };
    }
}
