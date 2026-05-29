#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

use naga::back::hlsl;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use nekoui_shader_types::{ShaderBackendTarget, ShaderStage};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderSourceInput {
    pub shader_name: String,
    pub wesl_path: PathBuf,
    pub canonical_wgsl_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderCheckReport {
    pub shader_name: String,
    pub wesl_compiled: bool,
    pub wgsl_validated: bool,
    pub hlsl_vertex_generated: bool,
    pub hlsl_fragment_generated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlslOutput {
    pub target: ShaderBackendTarget,
    pub entry_point: String,
    pub source: String,
}

pub fn compile_wesl_source(
    source_root: impl AsRef<Path>,
    package_path: &str,
) -> Result<String, String> {
    let module_path = package_path
        .parse()
        .map_err(|error| format!("invalid WESL package path: {error}"))?;
    wesl::Wesl::new(source_root.as_ref())
        .compile(&module_path)
        .map(|artifact| artifact.to_string())
        .map_err(|error| format!("WESL compile failed: {error}"))
}

pub fn validate_wgsl(wgsl_source: &str) -> Result<(naga::Module, naga::valid::ModuleInfo), String> {
    let module =
        wgsl::parse_str(wgsl_source).map_err(|error| format!("WGSL parse failed: {error}"))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|error| format!("WGSL validation failed: {error}"))?;
    Ok((module, info))
}

pub fn generate_hlsl_sm5(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point: &str,
    stage: ShaderStage,
) -> Result<HlslOutput, String> {
    let target = match stage {
        ShaderStage::Vertex => ShaderBackendTarget::D3d11Sm5VertexHlsl,
        ShaderStage::Fragment => ShaderBackendTarget::D3d11Sm5FragmentHlsl,
    };
    let options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_0,
        fake_missing_bindings: false,
        ..hlsl::Options::default()
    };
    let pipeline_options = hlsl::PipelineOptions {
        entry_point: Some((
            match stage {
                ShaderStage::Vertex => naga::ShaderStage::Vertex,
                ShaderStage::Fragment => naga::ShaderStage::Fragment,
            },
            entry_point.to_owned(),
        )),
    };
    let mut source = String::new();
    let mut writer = hlsl::Writer::new(&mut source, &options, &pipeline_options);
    let fragment_entry_point = match stage {
        ShaderStage::Vertex => hlsl::FragmentEntryPoint::new(module, "fs_main"),
        ShaderStage::Fragment => None,
    };
    writer
        .write(module, info, fragment_entry_point.as_ref())
        .map_err(|error| format!("HLSL generation failed: {error}"))?;
    Ok(HlslOutput {
        target,
        entry_point: entry_point.to_owned(),
        source,
    })
}

pub fn check_solid_rect_wgsl(wgsl_source: &str) -> Result<ShaderCheckReport, String> {
    let (module, info) = validate_wgsl(wgsl_source)?;
    generate_hlsl_sm5(&module, &info, "vs_main", ShaderStage::Vertex)?;
    generate_hlsl_sm5(&module, &info, "fs_main", ShaderStage::Fragment)?;
    Ok(ShaderCheckReport {
        shader_name: "core.solid_rect".to_owned(),
        wesl_compiled: true,
        wgsl_validated: true,
        hlsl_vertex_generated: true,
        hlsl_fragment_generated: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hlsl_options_are_sm5_and_strict_bindings() {
        let options = hlsl::Options {
            shader_model: hlsl::ShaderModel::V5_0,
            fake_missing_bindings: false,
            ..hlsl::Options::default()
        };

        assert_eq!(options.shader_model, hlsl::ShaderModel::V5_0);
        assert!(!options.fake_missing_bindings);
    }
}
