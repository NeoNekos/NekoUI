#![deny(unsafe_code)]

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CoreShaderId {
    SolidRect,
    GlyphMono,
}

impl CoreShaderId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SolidRect => "core.solid_rect",
            Self::GlyphMono => "core.glyph_mono",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ShaderBackendTarget {
    Wgsl,
    D3d11Sm5VertexHlsl,
    D3d11Sm5FragmentHlsl,
    D3d11Sm5VertexDxbc,
    D3d11Sm5FragmentDxbc,
}

impl ShaderBackendTarget {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wgsl => "wgsl",
            Self::D3d11Sm5VertexHlsl => "d3d11.sm5.vertex.hlsl",
            Self::D3d11Sm5FragmentHlsl => "d3d11.sm5.fragment.hlsl",
            Self::D3d11Sm5VertexDxbc => "d3d11.sm5.vertex.dxbc",
            Self::D3d11Sm5FragmentDxbc => "d3d11.sm5.fragment.dxbc",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl ShaderStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderEntryPoint {
    pub stage: ShaderStage,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VertexAttributeLayout {
    pub semantic: &'static str,
    pub semantic_index: u32,
    pub format: &'static str,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderArtifactTarget {
    pub target: ShaderBackendTarget,
    pub path: &'static str,
    pub sha256: &'static str,
    pub checked_binary: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoreShaderManifest {
    pub schema_version: u32,
    pub shader: CoreShaderId,
    pub source_path: &'static str,
    pub source_sha256: &'static str,
    pub canonical_wgsl_path: &'static str,
    pub canonical_wgsl_sha256: &'static str,
    pub entry_points: &'static [ShaderEntryPoint],
    pub targets: &'static [ShaderArtifactTarget],
    pub vertex_stride: u32,
    pub vertex_attributes: &'static [VertexAttributeLayout],
    pub notes: &'static str,
}
