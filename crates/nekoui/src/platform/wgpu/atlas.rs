use cosmic_text::CacheKey;
use etagere::{Allocation, AtlasAllocator, size2};
use hashbrown::HashMap;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Device, FilterMode, Origin3d,
    Queue, Sampler, SamplerDescriptor, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use crate::error::PlatformError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlyphAtlasKind {
    Mono,
    Color,
}

pub(crate) struct GlyphAtlas {
    kind: GlyphAtlasKind,
    allocator: AtlasAllocator,
    texture: Texture,
    _view: TextureView,
    _sampler: Sampler,
    bind_group: BindGroup,
    width: u32,
    height: u32,
    entries: HashMap<CacheKey, AtlasEntry>,
}

#[derive(Clone, Copy)]
pub(crate) struct AtlasEntry {
    _allocation: Allocation,
    pub(crate) placement_left: i32,
    pub(crate) placement_top: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) uv_rect: [f32; 4],
}

impl GlyphAtlas {
    pub(crate) fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        kind: GlyphAtlasKind,
        size: u32,
    ) -> Result<Self, PlatformError> {
        let size = size.max(1).min(u16::MAX as u32);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(match kind {
                GlyphAtlasKind::Mono => "nekoui_mono_glyph_atlas",
                GlyphAtlasKind::Color => "nekoui_color_glyph_atlas",
            }),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: match kind {
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
            label: Some(match kind {
                GlyphAtlasKind::Mono => "nekoui_mono_glyph_bind_group",
                GlyphAtlasKind::Color => "nekoui_color_glyph_bind_group",
            }),
            layout: bind_group_layout,
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

        Ok(Self {
            kind,
            allocator: AtlasAllocator::new(size2(size as i32, size as i32)),
            texture,
            _view: view,
            _sampler: sampler,
            bind_group,
            width: size,
            height: size,
            entries: HashMap::new(),
        })
    }

    pub(crate) fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    pub(crate) fn get(&self, key: &CacheKey) -> Option<AtlasEntry> {
        self.entries.get(key).copied()
    }

    pub(crate) fn upload_mask(
        &mut self,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
    ) -> Option<AtlasEntry> {
        debug_assert_eq!(self.kind, GlyphAtlasKind::Mono);
        self.upload_impl(queue, key, image, image.data.clone(), image.placement.width)
    }

    pub(crate) fn upload_color(
        &mut self,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
    ) -> Option<AtlasEntry> {
        debug_assert_eq!(self.kind, GlyphAtlasKind::Color);
        self.upload_impl(
            queue,
            key,
            image,
            image.data.clone(),
            image.placement.width * 4,
        )
    }

    fn upload_impl(
        &mut self,
        queue: &Queue,
        key: CacheKey,
        image: &cosmic_text::SwashImage,
        bytes: Vec<u8>,
        bytes_per_row: u32,
    ) -> Option<AtlasEntry> {
        if let Some(entry) = self.entries.get(&key).copied() {
            return Some(entry);
        }

        if image.placement.width == 0 || image.placement.height == 0 {
            return None;
        }

        let size = size2(image.placement.width as i32, image.placement.height as i32);
        let allocation = self.allocator.allocate(size).or_else(|| {
            self.allocator.clear();
            self.entries.clear();
            self.allocator.allocate(size)
        })?;

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.texture,
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
        self.entries.insert(key, entry);
        Some(entry)
    }
}
