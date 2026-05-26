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

    pub(crate) fn family(&self) -> &'static str {
        self.family
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
    pub(crate) fn generation(&self) -> FontGeneration {
        self.generation
    }

    pub(crate) fn default_family(&self) -> &'static str {
        self.entries
            .first()
            .map(FontMetadata::family)
            .unwrap_or("NekoUI deterministic fallback")
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
