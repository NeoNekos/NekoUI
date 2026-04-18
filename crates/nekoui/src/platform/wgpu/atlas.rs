use cosmic_text::CacheKey;
use etagere::{Allocation, AtlasAllocator, size2};
use rustc_hash::FxHashMap;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Device, FilterMode, Origin3d,
    Queue, Sampler, SamplerDescriptor, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use crate::error::PlatformError;

const MAX_ATLAS_PAGES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlyphAtlasKind {
    Mono,
    Color,
}

pub(crate) struct GlyphAtlas {
    kind: GlyphAtlasKind,
    bind_group_layout: BindGroupLayout,
    width: u32,
    height: u32,
    pages: Vec<AtlasPage>,
    entries: FxHashMap<CacheKey, AtlasEntry>,
    next_page_id: u32,
    frame_id: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct AtlasEntry {
    pub(crate) page_id: u32,
    _allocation: Allocation,
    pub(crate) placement_left: i32,
    pub(crate) placement_top: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) uv_rect: [f32; 4],
}

struct AtlasPage {
    id: u32,
    allocator: AtlasAllocator,
    texture: Texture,
    _view: TextureView,
    _sampler: Sampler,
    bind_group: BindGroup,
    entries: FxHashMap<CacheKey, AtlasEntry>,
    used_in_frame: bool,
    last_used_frame: u64,
}

impl GlyphAtlas {
    pub(crate) fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        kind: GlyphAtlasKind,
        size: u32,
    ) -> Result<Self, PlatformError> {
        let size = size.max(1).min(u16::MAX as u32);
        let mut atlas = Self {
            kind,
            bind_group_layout: bind_group_layout.clone(),
            width: size,
            height: size,
            pages: Vec::new(),
            entries: FxHashMap::default(),
            next_page_id: 0,
            frame_id: 0,
        };
        let first_page = atlas.create_page(device);
        atlas.pages.push(first_page);
        Ok(atlas)
    }

    pub(crate) fn begin_frame(&mut self) {
        self.frame_id = self.frame_id.saturating_add(1);
        for page in &mut self.pages {
            page.used_in_frame = false;
        }
    }

    pub(crate) fn bind_group(&self, page_id: u32) -> Option<&BindGroup> {
        self.pages
            .iter()
            .find(|page| page.id == page_id)
            .map(|page| &page.bind_group)
    }

    pub(crate) fn get(&mut self, key: &CacheKey) -> Option<AtlasEntry> {
        let entry = self.entries.get(key).copied()?;
        self.mark_page_used(entry.page_id);
        Some(entry)
    }

    pub(crate) fn upload_mask(
        &mut self,
        device: &Device,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
    ) -> Option<AtlasEntry> {
        debug_assert_eq!(self.kind, GlyphAtlasKind::Mono);
        self.upload_impl(
            device,
            queue,
            key,
            image,
            image.data.clone(),
            image.placement.width,
        )
    }

    pub(crate) fn upload_color(
        &mut self,
        device: &Device,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
    ) -> Option<AtlasEntry> {
        debug_assert_eq!(self.kind, GlyphAtlasKind::Color);
        self.upload_impl(
            device,
            queue,
            key,
            image,
            image.data.clone(),
            image.placement.width * 4,
        )
    }

    fn upload_impl(
        &mut self,
        device: &Device,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
        bytes: Vec<u8>,
        bytes_per_row: u32,
    ) -> Option<AtlasEntry> {
        if let Some(entry) = self.entries.get(&key).copied() {
            self.mark_page_used(entry.page_id);
            return Some(entry);
        }

        if image.placement.width == 0 || image.placement.height == 0 {
            return None;
        }

        let size = size2(image.placement.width as i32, image.placement.height as i32);
        let (page_index, allocation) = self.allocate_page_region(device, size)?;
        let page_id = self.pages[page_index].id;

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.pages[page_index].texture,
                mip_level: 0,
                origin: Origin3d {
                    x: allocation.rectangle.min.x as u32,
                    y: allocation.rectangle.min.y as u32,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            &bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(image.placement.height),
            },
            wgpu::Extent3d {
                width: image.placement.width,
                height: image.placement.height,
                depth_or_array_layers: 1,
            },
        );

        let atlas_width = self.width as f32;
        let atlas_height = self.height as f32;
        let entry = AtlasEntry {
            page_id,
            _allocation: allocation,
            placement_left: image.placement.left,
            placement_top: image.placement.top,
            width: image.placement.width,
            height: image.placement.height,
            uv_rect: [
                allocation.rectangle.min.x as f32 / atlas_width,
                allocation.rectangle.min.y as f32 / atlas_height,
                image.placement.width as f32 / atlas_width,
                image.placement.height as f32 / atlas_height,
            ],
        };
        self.pages[page_index].entries.insert(key, entry);
        self.pages[page_index].used_in_frame = true;
        self.pages[page_index].last_used_frame = self.frame_id;
        self.entries.insert(key, entry);
        Some(entry)
    }

    fn create_page(&mut self, device: &Device) -> AtlasPage {
        let page_id = self.next_page_id;
        self.next_page_id += 1;

        let texture = device.create_texture(&TextureDescriptor {
            label: Some(match self.kind {
                GlyphAtlasKind::Mono => "nekoui_mono_glyph_atlas",
                GlyphAtlasKind::Color => "nekoui_color_glyph_atlas",
            }),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: match self.kind {
                GlyphAtlasKind::Mono => TextureFormat::R8Unorm,
                GlyphAtlasKind::Color => TextureFormat::Rgba8Unorm,
            },
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("nekoui_glyph_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(match self.kind {
                GlyphAtlasKind::Mono => "nekoui_mono_glyph_bind_group",
                GlyphAtlasKind::Color => "nekoui_color_glyph_bind_group",
            }),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        AtlasPage {
            id: page_id,
            allocator: AtlasAllocator::new(size2(self.width as i32, self.height as i32)),
            texture,
            _view: view,
            _sampler: sampler,
            bind_group,
            entries: FxHashMap::default(),
            used_in_frame: false,
            last_used_frame: self.frame_id,
        }
    }

    fn allocate_page_region(
        &mut self,
        device: &Device,
        size: etagere::Size,
    ) -> Option<(usize, Allocation)> {
        if self.pages.is_empty() {
            let page = self.create_page(device);
            self.pages.push(page);
        }

        let current_index = self.pages.len() - 1;
        if let Some(allocation) = self.pages[current_index].allocator.allocate(size) {
            return Some((current_index, allocation));
        }

        if self.pages.len() < MAX_ATLAS_PAGES {
            let page = self.create_page(device);
            self.pages.push(page);
            let current_index = self.pages.len() - 1;
            return self.pages[current_index]
                .allocator
                .allocate(size)
                .map(|allocation| (current_index, allocation));
        }

        self.evict_unused_pages();

        if self.pages.is_empty() {
            let page = self.create_page(device);
            self.pages.push(page);
            return self.pages[0]
                .allocator
                .allocate(size)
                .map(|allocation| (0, allocation));
        }

        let current_index = self.pages.len() - 1;
        if let Some(allocation) = self.pages[current_index].allocator.allocate(size) {
            return Some((current_index, allocation));
        }

        if self.pages.len() < MAX_ATLAS_PAGES {
            let page = self.create_page(device);
            self.pages.push(page);
            let current_index = self.pages.len() - 1;
            return self.pages[current_index]
                .allocator
                .allocate(size)
                .map(|allocation| (current_index, allocation));
        }

        None
    }

    fn evict_unused_pages(&mut self) {
        if self.pages.len() < MAX_ATLAS_PAGES {
            return;
        }

        let eviction_ids = eviction_page_ids(
            &self
                .pages
                .iter()
                .map(|page| AtlasPageState {
                    id: page.id,
                    used_in_frame: page.used_in_frame,
                    last_used_frame: page.last_used_frame,
                })
                .collect::<Vec<_>>(),
            MAX_ATLAS_PAGES,
        );

        if eviction_ids.is_empty() {
            return;
        }

        for page_id in &eviction_ids {
            if let Some(page) = self.pages.iter().find(|page| page.id == *page_id) {
                for key in page.entries.keys().copied().collect::<Vec<_>>() {
                    self.entries.remove(&key);
                }
            }
        }

        self.pages.retain(|page| !eviction_ids.contains(&page.id));
    }

    fn mark_page_used(&mut self, page_id: u32) {
        if let Some(page) = self.pages.iter_mut().find(|page| page.id == page_id) {
            page.used_in_frame = true;
            page.last_used_frame = self.frame_id;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AtlasPageState {
    id: u32,
    used_in_frame: bool,
    last_used_frame: u64,
}

fn eviction_page_ids(pages: &[AtlasPageState], max_pages: usize) -> Vec<u32> {
    if pages.len() < max_pages {
        return Vec::new();
    }

    let removable = pages.len().saturating_sub(max_pages - 1);
    let mut candidates = pages
        .iter()
        .copied()
        .filter(|page| !page.used_in_frame)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|page| page.last_used_frame);
    candidates
        .into_iter()
        .take(removable)
        .map(|page| page.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{AtlasPageState, MAX_ATLAS_PAGES, eviction_page_ids};

    #[test]
    fn eviction_prefers_oldest_unused_pages() {
        let eviction = eviction_page_ids(
            &[
                AtlasPageState {
                    id: 1,
                    used_in_frame: true,
                    last_used_frame: 9,
                },
                AtlasPageState {
                    id: 2,
                    used_in_frame: false,
                    last_used_frame: 3,
                },
                AtlasPageState {
                    id: 3,
                    used_in_frame: false,
                    last_used_frame: 7,
                },
                AtlasPageState {
                    id: 4,
                    used_in_frame: false,
                    last_used_frame: 5,
                },
            ],
            MAX_ATLAS_PAGES,
        );

        assert_eq!(eviction, vec![2]);
    }

    #[test]
    fn eviction_keeps_used_pages_even_when_full() {
        let eviction = eviction_page_ids(
            &[
                AtlasPageState {
                    id: 1,
                    used_in_frame: true,
                    last_used_frame: 9,
                },
                AtlasPageState {
                    id: 2,
                    used_in_frame: true,
                    last_used_frame: 8,
                },
                AtlasPageState {
                    id: 3,
                    used_in_frame: true,
                    last_used_frame: 7,
                },
                AtlasPageState {
                    id: 4,
                    used_in_frame: true,
                    last_used_frame: 6,
                },
            ],
            MAX_ATLAS_PAGES,
        );

        assert!(eviction.is_empty());
    }
}
