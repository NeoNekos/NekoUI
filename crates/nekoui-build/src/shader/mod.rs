use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

mod discovery;
mod dxbc;
mod hlsl;
mod metadata;
mod rust_module;

const GENERATED_MODULE: &str = "shaders.rs";
const SHADER_ARTIFACT_DIR: &str = "shaders";
const DXBC_ENV_OVERRIDES: &[&str] = &["NEKOUI_FXC"];

#[derive(Clone, Debug)]
pub(crate) struct ShaderDescriptor {
    pub(crate) logical_id: String,
    pub(crate) file_stem: String,
    pub(crate) variant_name: String,
    pub(crate) constant_prefix: String,
    pub(crate) source_file: String,
    pub(crate) wesl_package_path: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShaderStage {
    Vertex,
    Fragment,
}

impl ShaderStage {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }

    pub(crate) const fn naga_stage(self) -> naga::ShaderStage {
        match self {
            Self::Vertex => naga::ShaderStage::Vertex,
            Self::Fragment => naga::ShaderStage::Fragment,
        }
    }

    const fn hlsl_profile(self) -> &'static str {
        match self {
            Self::Vertex => "vs_5_0",
            Self::Fragment => "ps_5_0",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EntryPointMetadata {
    pub(crate) stage: ShaderStage,
    pub(crate) name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct VertexAttributeMetadata {
    pub(crate) const_name: String,
    pub(crate) semantic: &'static str,
    pub(crate) semantic_index: u32,
    pub(crate) format: &'static str,
    pub(crate) offset: u32,
    pub(crate) byte_width: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum D3d11RegisterClass {
    Srv,
    Sampler,
    Cbv,
    Uav,
}

impl D3d11RegisterClass {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Srv => "srv",
            Self::Sampler => "sampler",
            Self::Cbv => "cbv",
            Self::Uav => "uav",
        }
    }

    pub(crate) const fn slot_const_suffix(self) -> &'static str {
        match self {
            Self::Srv => "SRV_SLOT",
            Self::Sampler => "SAMPLER_SLOT",
            Self::Cbv => "CBV_SLOT",
            Self::Uav => "UAV_SLOT",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct D3d11ResourceMetadata {
    pub(crate) name: String,
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) register_class: D3d11RegisterClass,
    pub(crate) slot: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct ShaderMetadata {
    pub(crate) entry_points: Vec<EntryPointMetadata>,
    pub(crate) vertex_stride: u32,
    pub(crate) vertex_attributes: Vec<VertexAttributeMetadata>,
    pub(crate) d3d11_resources: Vec<D3d11ResourceMetadata>,
}

impl ShaderMetadata {
    pub(crate) fn entry_point(&self, stage: ShaderStage) -> Result<&EntryPointMetadata, String> {
        self.entry_points
            .iter()
            .find(|entry_point| entry_point.stage == stage)
            .ok_or_else(|| format!("missing reflected {} entry point", stage.label()))
    }
}

#[derive(Debug)]
pub(crate) struct ShaderArtifacts {
    pub(crate) descriptor: ShaderDescriptor,
    pub(crate) metadata: ShaderMetadata,
    pub(crate) wgsl_path: PathBuf,
    pub(crate) hlsl_vertex_path: PathBuf,
    pub(crate) hlsl_fragment_path: PathBuf,
    pub(crate) dxbc_vertex_path: Option<PathBuf>,
    pub(crate) dxbc_fragment_path: Option<PathBuf>,
}

pub fn build_shaders() -> Result<(), String> {
    let manifest_dir = path_from_env("CARGO_MANIFEST_DIR")?;
    let source_dir = manifest_dir.join("src").join("platform").join("shader");
    let out_dir = path_from_env("OUT_DIR")?;

    println!("cargo:rerun-if-changed={}", source_dir.display());
    for override_name in DXBC_ENV_OVERRIDES {
        println!("cargo:rerun-if-env-changed={override_name}");
    }

    let compiler = if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        Some(dxbc::find_dxbc_compiler()?)
    } else {
        None
    };
    let shader_out_dir = out_dir.join(SHADER_ARTIFACT_DIR);
    fs::create_dir_all(&shader_out_dir).map_err(|error| {
        format!(
            "failed to create shader output directory {}: {error}",
            shader_out_dir.display()
        )
    })?;

    let descriptors = discovery::discover_shaders(&source_dir)?;
    let mut artifacts = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        artifacts.push(build_shader(
            descriptor,
            &source_dir,
            &shader_out_dir,
            compiler.as_ref(),
        )?);
    }

    rust_module::write_generated_module(&artifacts, &out_dir.join(GENERATED_MODULE))?;
    Ok(())
}

fn path_from_env(name: &str) -> Result<PathBuf, String> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("environment variable {name} is not set"))
}

fn build_shader(
    descriptor: ShaderDescriptor,
    source_dir: &Path,
    out_dir: &Path,
    compiler: Option<&dxbc::Compiler>,
) -> Result<ShaderArtifacts, String> {
    let source_path = source_dir.join(&descriptor.source_file);
    println!("cargo:rerun-if-changed={}", source_path.display());

    let wgsl_source = compile_wesl_source(source_dir, &descriptor)?;
    let (module, info) = validate_wgsl(&descriptor, &wgsl_source)?;
    let metadata = metadata::reflect_shader(&descriptor, &module)?;
    let vertex_hlsl =
        hlsl::generate_hlsl_sm5(&descriptor, &metadata, &module, &info, ShaderStage::Vertex)?;
    let fragment_hlsl = hlsl::generate_hlsl_sm5(
        &descriptor,
        &metadata,
        &module,
        &info,
        ShaderStage::Fragment,
    )?;

    let wgsl_path = out_dir.join(format!("{}.wgsl", descriptor.file_stem));
    let hlsl_vertex_path = out_dir.join(format!("{}.vs_5_0.hlsl", descriptor.file_stem));
    let hlsl_fragment_path = out_dir.join(format!("{}.ps_5_0.hlsl", descriptor.file_stem));
    let dxbc_vertex_path = out_dir.join(format!("{}.vs_5_0.cso", descriptor.file_stem));
    let dxbc_fragment_path = out_dir.join(format!("{}.ps_5_0.cso", descriptor.file_stem));

    write_text(&wgsl_path, &wgsl_source)?;
    write_text(&hlsl_vertex_path, &vertex_hlsl.source)?;
    write_text(&hlsl_fragment_path, &fragment_hlsl.source)?;
    let (dxbc_vertex_path, dxbc_fragment_path) = if let Some(compiler) = compiler {
        dxbc::compile_dxbc(
            compiler,
            &descriptor,
            &hlsl_vertex_path,
            &vertex_hlsl.entry_point,
            ShaderStage::Vertex,
            &dxbc_vertex_path,
        )?;
        dxbc::compile_dxbc(
            compiler,
            &descriptor,
            &hlsl_fragment_path,
            &fragment_hlsl.entry_point,
            ShaderStage::Fragment,
            &dxbc_fragment_path,
        )?;
        (Some(dxbc_vertex_path), Some(dxbc_fragment_path))
    } else {
        (None, None)
    };

    Ok(ShaderArtifacts {
        descriptor,
        metadata,
        wgsl_path,
        hlsl_vertex_path,
        hlsl_fragment_path,
        dxbc_vertex_path,
        dxbc_fragment_path,
    })
}

fn compile_wesl_source(source_dir: &Path, descriptor: &ShaderDescriptor) -> Result<String, String> {
    let module_path = descriptor.wesl_package_path.parse().map_err(|error| {
        format!(
            "{} invalid WESL package path: {error}",
            descriptor.logical_id
        )
    })?;
    wesl::Wesl::new(source_dir)
        .compile(&module_path)
        .map(|artifact| artifact.to_string())
        .map_err(|error| format!("{} WESL compile failed: {error}", descriptor.logical_id))
}

fn validate_wgsl(
    descriptor: &ShaderDescriptor,
    wgsl_source: &str,
) -> Result<(naga::Module, naga::valid::ModuleInfo), String> {
    let module = wgsl::parse_str(wgsl_source)
        .map_err(|error| format!("{} WGSL parse failed: {error}", descriptor.logical_id))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|error| format!("{} WGSL validation failed: {error}", descriptor.logical_id))?;
    Ok((module, info))
}

fn write_text(path: &Path, contents: &str) -> Result<(), String> {
    fs::write(path, contents).map_err(|error| {
        format!(
            "failed to write shader artifact {}: {error}",
            path.display()
        )
    })
}

pub(crate) fn is_ascii_snake_case(name: &str) -> bool {
    if name.is_empty() || !name.is_ascii() {
        return false;
    }
    let mut previous_underscore = false;
    for (index, byte) in name.bytes().enumerate() {
        match byte {
            b'a'..=b'z' => previous_underscore = false,
            b'0'..=b'9' if index > 0 => previous_underscore = false,
            b'_' if index > 0 && !previous_underscore => previous_underscore = true,
            _ => return false,
        }
    }
    !previous_underscore
}

pub(crate) fn constant_name_from_snake_case(name: &str) -> String {
    name.to_ascii_uppercase()
}
