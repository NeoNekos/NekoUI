#[cfg(test)]
use nekoui_shader_types::{
    CoreShaderId, CoreShaderManifest, ShaderArtifactTarget, ShaderBackendTarget, ShaderEntryPoint,
    ShaderStage, VertexAttributeLayout,
};

#[cfg(test)]
pub(crate) const SOLID_RECT_VERTEX_STRIDE: u32 = 24;
#[cfg(any(test, target_os = "windows"))]
pub(crate) const SOLID_RECT_POSITION_OFFSET: u32 = 0;
#[cfg(any(test, target_os = "windows"))]
pub(crate) const SOLID_RECT_COLOR_OFFSET: u32 = 8;
#[cfg(test)]
pub(crate) const GLYPH_MONO_VERTEX_STRIDE: u32 = 32;
#[cfg(any(test, target_os = "windows"))]
pub(crate) const GLYPH_MONO_POSITION_OFFSET: u32 = 0;
#[cfg(any(test, target_os = "windows"))]
pub(crate) const GLYPH_MONO_UV_OFFSET: u32 = 8;
#[cfg(any(test, target_os = "windows"))]
pub(crate) const GLYPH_MONO_COLOR_OFFSET: u32 = 16;

// TODO: 调整管线路径，寻找更优雅的shader管线
#[cfg(test)]
pub(crate) const SOLID_RECT_WGSL: &str = include_str!("../../shaders/generated/solid_rect.wgsl");
#[cfg(test)]
pub(crate) const SOLID_RECT_D3D11_VERTEX_HLSL: &str =
    include_str!("../../shaders/generated/solid_rect.vs_5_0.hlsl");
#[cfg(test)]
pub(crate) const SOLID_RECT_D3D11_FRAGMENT_HLSL: &str =
    include_str!("../../shaders/generated/solid_rect.ps_5_0.hlsl");
#[cfg(any(test, target_os = "windows"))]
pub(crate) const SOLID_RECT_D3D11_VERTEX_DXBC: &[u8] =
    include_bytes!("../../shaders/artifacts/solid_rect.vs_5_0.cso");
#[cfg(any(test, target_os = "windows"))]
pub(crate) const SOLID_RECT_D3D11_FRAGMENT_DXBC: &[u8] =
    include_bytes!("../../shaders/artifacts/solid_rect.ps_5_0.cso");
#[cfg(test)]
pub(crate) const GLYPH_MONO_WGSL: &str = include_str!("../../shaders/generated/glyph_mono.wgsl");
#[cfg(test)]
pub(crate) const GLYPH_MONO_D3D11_VERTEX_HLSL: &str =
    include_str!("../../shaders/generated/glyph_mono.vs_5_0.hlsl");
#[cfg(test)]
pub(crate) const GLYPH_MONO_D3D11_FRAGMENT_HLSL: &str =
    include_str!("../../shaders/generated/glyph_mono.ps_5_0.hlsl");
#[cfg(any(test, target_os = "windows"))]
pub(crate) const GLYPH_MONO_D3D11_VERTEX_DXBC: &[u8] =
    include_bytes!("../../shaders/artifacts/glyph_mono.vs_5_0.cso");
#[cfg(any(test, target_os = "windows"))]
pub(crate) const GLYPH_MONO_D3D11_FRAGMENT_DXBC: &[u8] =
    include_bytes!("../../shaders/artifacts/glyph_mono.ps_5_0.cso");

#[cfg(test)]
pub(crate) const SOLID_RECT_ENTRY_POINTS: &[ShaderEntryPoint] = &[
    ShaderEntryPoint {
        stage: ShaderStage::Vertex,
        name: "vs_main",
    },
    ShaderEntryPoint {
        stage: ShaderStage::Fragment,
        name: "fs_main",
    },
];

#[cfg(test)]
pub(crate) const SOLID_RECT_VERTEX_LAYOUT: &[VertexAttributeLayout] = &[
    VertexAttributeLayout {
        semantic: "POSITION",
        semantic_index: 0,
        format: "R32G32_FLOAT",
        offset: SOLID_RECT_POSITION_OFFSET,
    },
    VertexAttributeLayout {
        semantic: "COLOR",
        semantic_index: 0,
        format: "R32G32B32A32_FLOAT",
        offset: SOLID_RECT_COLOR_OFFSET,
    },
];

#[cfg(test)]
pub(crate) const GLYPH_MONO_ENTRY_POINTS: &[ShaderEntryPoint] = SOLID_RECT_ENTRY_POINTS;

#[cfg(test)]
pub(crate) const GLYPH_MONO_VERTEX_LAYOUT: &[VertexAttributeLayout] = &[
    VertexAttributeLayout {
        semantic: "POSITION",
        semantic_index: 0,
        format: "R32G32_FLOAT",
        offset: GLYPH_MONO_POSITION_OFFSET,
    },
    VertexAttributeLayout {
        semantic: "TEXCOORD",
        semantic_index: 0,
        format: "R32G32_FLOAT",
        offset: GLYPH_MONO_UV_OFFSET,
    },
    VertexAttributeLayout {
        semantic: "COLOR",
        semantic_index: 0,
        format: "R32G32B32A32_FLOAT",
        offset: GLYPH_MONO_COLOR_OFFSET,
    },
];

#[cfg(test)]
pub(crate) const SOLID_RECT_TARGETS: &[ShaderArtifactTarget] = &[
    ShaderArtifactTarget {
        target: ShaderBackendTarget::Wgsl,
        path: "shaders/generated/solid_rect.wgsl",
        sha256: "42522acb26710c062681ce50507c96c32538eefc32fc178ba2d4a2bc64698709",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5VertexHlsl,
        path: "shaders/generated/solid_rect.vs_5_0.hlsl",
        sha256: "d4f04447fb2e5eb26371be3a7e0db823506fd18772146d079100f2faee1d25ef",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5FragmentHlsl,
        path: "shaders/generated/solid_rect.ps_5_0.hlsl",
        sha256: "23f5bdb944ebbd4601cb5266f882d09fe30414600eafca1713338441e61a9eca",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5VertexDxbc,
        path: "shaders/artifacts/solid_rect.vs_5_0.cso",
        sha256: "e83ea7f63eb46db6bbc47ab9112482a8452545181454de0543f0ee1d4b4c6009",
        checked_binary: true,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5FragmentDxbc,
        path: "shaders/artifacts/solid_rect.ps_5_0.cso",
        sha256: "0553edf2d013f5b6adc6493816537867e526a9f10024fbd2a649f7146187a186",
        checked_binary: true,
    },
];

#[cfg(test)]
pub(crate) const GLYPH_MONO_TARGETS: &[ShaderArtifactTarget] = &[
    ShaderArtifactTarget {
        target: ShaderBackendTarget::Wgsl,
        path: "shaders/generated/glyph_mono.wgsl",
        sha256: "1330ef1fc6677ed5d94bc976e6125c4e7165a5a7bc5ce9d41af48c925f3f9600",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5VertexHlsl,
        path: "shaders/generated/glyph_mono.vs_5_0.hlsl",
        sha256: "6fc414a5ba8e69df041d0873e1ae5eed6be22c2469b2bff005e77f6141aa5aed",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5FragmentHlsl,
        path: "shaders/generated/glyph_mono.ps_5_0.hlsl",
        sha256: "0982f0463a2354f7a9d6754a758ecc8d48cb56e7f6044b7e36e6e699eda7a1ac",
        checked_binary: false,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5VertexDxbc,
        path: "shaders/artifacts/glyph_mono.vs_5_0.cso",
        sha256: "047cc61306e18b9c595cb759a7503a1199d492694b279796b95ded3119305563",
        checked_binary: true,
    },
    ShaderArtifactTarget {
        target: ShaderBackendTarget::D3d11Sm5FragmentDxbc,
        path: "shaders/artifacts/glyph_mono.ps_5_0.cso",
        sha256: "1319075da6d24b0b6781339b2d94d7871bf10545b942b50f91d2d49adf98c028",
        checked_binary: true,
    },
];

#[cfg(test)]
pub(crate) const SOLID_RECT_MANIFEST: CoreShaderManifest = CoreShaderManifest {
    schema_version: 1,
    shader: CoreShaderId::SolidRect,
    source_path: "shaders/framework/solid_rect.wesl",
    source_sha256: "42522acb26710c062681ce50507c96c32538eefc32fc178ba2d4a2bc64698709",
    canonical_wgsl_path: "shaders/generated/solid_rect.wgsl",
    canonical_wgsl_sha256: "42522acb26710c062681ce50507c96c32538eefc32fc178ba2d4a2bc64698709",
    entry_points: SOLID_RECT_ENTRY_POINTS,
    targets: SOLID_RECT_TARGETS,
    vertex_stride: SOLID_RECT_VERTEX_STRIDE,
    vertex_attributes: SOLID_RECT_VERTEX_LAYOUT,
    notes: "Runtime loads checked artifacts and never invokes WESL or Naga for core shaders.",
};

#[cfg(test)]
pub(crate) const GLYPH_MONO_MANIFEST: CoreShaderManifest = CoreShaderManifest {
    schema_version: 1,
    shader: CoreShaderId::GlyphMono,
    source_path: "shaders/framework/glyph_mono.wesl",
    source_sha256: "1330ef1fc6677ed5d94bc976e6125c4e7165a5a7bc5ce9d41af48c925f3f9600",
    canonical_wgsl_path: "shaders/generated/glyph_mono.wgsl",
    canonical_wgsl_sha256: "1330ef1fc6677ed5d94bc976e6125c4e7165a5a7bc5ce9d41af48c925f3f9600",
    entry_points: GLYPH_MONO_ENTRY_POINTS,
    targets: GLYPH_MONO_TARGETS,
    vertex_stride: GLYPH_MONO_VERTEX_STRIDE,
    vertex_attributes: GLYPH_MONO_VERTEX_LAYOUT,
    notes: "Runtime loads checked artifacts and never invokes WESL or Naga for core shaders.",
};
