use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use super::{D3d11ResourceMetadata, ShaderArtifacts, VertexAttributeMetadata};

pub(crate) fn write_generated_module(
    artifacts: &[ShaderArtifacts],
    module_path: &Path,
) -> Result<(), String> {
    let mut module = String::new();
    module.push_str("#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]\n");
    module.push_str("pub(crate) enum CoreShader {\n");
    for artifact in artifacts {
        writeln!(module, "    {},", artifact.descriptor.variant_name).unwrap();
    }
    module.push_str("}\n\n");

    module.push_str("#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n");
    module.push_str("pub(crate) struct EntryPoint {\n");
    module.push_str("    pub(crate) stage: &'static str,\n");
    module.push_str("    pub(crate) name: &'static str,\n");
    module.push_str("}\n\n");

    module.push_str("#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n");
    module.push_str("pub(crate) struct VertexAttribute {\n");
    module.push_str("    pub(crate) semantic: &'static str,\n");
    module.push_str("    pub(crate) semantic_index: u32,\n");
    module.push_str("    pub(crate) format: &'static str,\n");
    module.push_str("    pub(crate) offset: u32,\n");
    module.push_str("}\n\n");

    module.push_str("#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n");
    module.push_str("pub(crate) struct D3d11ResourceBinding {\n");
    module.push_str("    pub(crate) name: &'static str,\n");
    module.push_str("    pub(crate) group: u32,\n");
    module.push_str("    pub(crate) binding: u32,\n");
    module.push_str("    pub(crate) register_class: &'static str,\n");
    module.push_str("    pub(crate) slot: u32,\n");
    module.push_str("}\n\n");

    module.push_str("#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n");
    module.push_str("pub(crate) struct CoreShaderArtifacts {\n");
    module.push_str("    pub(crate) id: CoreShader,\n");
    module.push_str("    pub(crate) name: &'static str,\n");
    module.push_str("    pub(crate) wgsl: &'static str,\n");
    module.push_str("    pub(crate) d3d11_vertex_hlsl: &'static str,\n");
    module.push_str("    pub(crate) d3d11_fragment_hlsl: &'static str,\n");
    module.push_str("    pub(crate) d3d11_vertex_dxbc: &'static [u8],\n");
    module.push_str("    pub(crate) d3d11_fragment_dxbc: &'static [u8],\n");
    module.push_str("    pub(crate) vertex_stride: u32,\n");
    module.push_str("    pub(crate) entry_points: &'static [EntryPoint],\n");
    module.push_str("    pub(crate) vertex_attributes: &'static [VertexAttribute],\n");
    module.push_str("    pub(crate) d3d11_resource_bindings: &'static [D3d11ResourceBinding],\n");
    module.push_str("}\n\n");

    for artifact in artifacts {
        write_shader_constants(&mut module, artifact)?;
    }
    write_shader_registry(&mut module, artifacts)?;

    fs::write(module_path, module).map_err(|error| {
        format!(
            "failed to write generated shader Rust module {}: {error}",
            module_path.display()
        )
    })
}

fn write_shader_constants(module: &mut String, artifact: &ShaderArtifacts) -> Result<(), String> {
    let descriptor = &artifact.descriptor;
    let metadata = &artifact.metadata;
    let prefix = &descriptor.constant_prefix;
    writeln!(
        module,
        "pub(crate) const {prefix}_WGSL: &str = include_str!(r#\"{}\"#);",
        artifact.wgsl_path.display()
    )
    .unwrap();
    writeln!(
        module,
        "pub(crate) const {prefix}_D3D11_VERTEX_HLSL: &str = include_str!(r#\"{}\"#);",
        artifact.hlsl_vertex_path.display()
    )
    .unwrap();
    writeln!(
        module,
        "pub(crate) const {prefix}_D3D11_FRAGMENT_HLSL: &str = include_str!(r#\"{}\"#);",
        artifact.hlsl_fragment_path.display()
    )
    .unwrap();
    write_dxbc_constant(
        module,
        prefix,
        "VERTEX",
        artifact.dxbc_vertex_path.as_deref(),
    );
    write_dxbc_constant(
        module,
        prefix,
        "FRAGMENT",
        artifact.dxbc_fragment_path.as_deref(),
    );
    writeln!(
        module,
        "pub(crate) const {prefix}_VERTEX_STRIDE: u32 = {};",
        metadata.vertex_stride
    )
    .unwrap();
    for attribute in &metadata.vertex_attributes {
        write_vertex_offset_constant(module, prefix, attribute);
    }
    for resource in &metadata.d3d11_resources {
        write_resource_slot_constant(module, prefix, resource);
    }
    write_entry_points(module, prefix, artifact);
    write_vertex_attributes(module, prefix, &metadata.vertex_attributes);
    write_resource_bindings(module, prefix, &metadata.d3d11_resources);
    Ok(())
}

fn write_dxbc_constant(module: &mut String, prefix: &str, stage: &str, path: Option<&Path>) {
    if let Some(path) = path {
        writeln!(
            module,
            "pub(crate) const {prefix}_D3D11_{stage}_DXBC: &[u8] = include_bytes!(r#\"{}\"#);",
            path.display()
        )
        .unwrap();
    } else {
        writeln!(
            module,
            "pub(crate) const {prefix}_D3D11_{stage}_DXBC: &[u8] = &[];"
        )
        .unwrap();
    }
}

fn write_vertex_offset_constant(
    module: &mut String,
    prefix: &str,
    attribute: &VertexAttributeMetadata,
) {
    writeln!(
        module,
        "pub(crate) const {prefix}_{}_OFFSET: u32 = {};",
        attribute.const_name, attribute.offset
    )
    .unwrap();
}

fn write_resource_slot_constant(
    module: &mut String,
    prefix: &str,
    resource: &D3d11ResourceMetadata,
) {
    writeln!(
        module,
        "pub(crate) const {prefix}_{}_D3D11_{}: u32 = {};",
        resource.name.to_ascii_uppercase(),
        resource.register_class.slot_const_suffix(),
        resource.slot
    )
    .unwrap();
}

fn write_entry_points(module: &mut String, prefix: &str, artifact: &ShaderArtifacts) {
    writeln!(
        module,
        "pub(crate) const {prefix}_ENTRY_POINTS: &[EntryPoint] = &["
    )
    .unwrap();
    for entry_point in &artifact.metadata.entry_points {
        writeln!(
            module,
            "    EntryPoint {{ stage: \"{}\", name: \"{}\" }},",
            entry_point.stage.label(),
            entry_point.name
        )
        .unwrap();
    }
    module.push_str("];\n");
}

fn write_vertex_attributes(
    module: &mut String,
    prefix: &str,
    attributes: &[VertexAttributeMetadata],
) {
    writeln!(
        module,
        "pub(crate) const {prefix}_VERTEX_ATTRIBUTES: &[VertexAttribute] = &["
    )
    .unwrap();
    for attribute in attributes {
        writeln!(
            module,
            "    VertexAttribute {{ semantic: \"{}\", semantic_index: {}, format: \"{}\", offset: {} }},",
            attribute.semantic, attribute.semantic_index, attribute.format, attribute.offset,
        )
        .unwrap();
    }
    module.push_str("];\n");
}

fn write_resource_bindings(module: &mut String, prefix: &str, resources: &[D3d11ResourceMetadata]) {
    writeln!(
        module,
        "pub(crate) const {prefix}_D3D11_RESOURCE_BINDINGS: &[D3d11ResourceBinding] = &["
    )
    .unwrap();
    for resource in resources {
        writeln!(
            module,
            "    D3d11ResourceBinding {{ name: \"{}\", group: {}, binding: {}, register_class: \"{}\", slot: {} }},",
            resource.name,
            resource.group,
            resource.binding,
            resource.register_class.label(),
            resource.slot,
        )
        .unwrap();
    }
    module.push_str("];\n\n");
}

fn write_shader_registry(module: &mut String, artifacts: &[ShaderArtifacts]) -> Result<(), String> {
    module.push_str("pub(crate) const CORE_SHADERS: &[CoreShaderArtifacts] = &[\n");
    for artifact in artifacts {
        let descriptor = &artifact.descriptor;
        let prefix = &descriptor.constant_prefix;
        writeln!(module, "    CoreShaderArtifacts {{").unwrap();
        writeln!(
            module,
            "        id: CoreShader::{},",
            descriptor.variant_name
        )
        .unwrap();
        writeln!(module, "        name: \"{}\",", descriptor.logical_id).unwrap();
        writeln!(module, "        wgsl: {prefix}_WGSL,").unwrap();
        writeln!(
            module,
            "        d3d11_vertex_hlsl: {prefix}_D3D11_VERTEX_HLSL,"
        )
        .unwrap();
        writeln!(
            module,
            "        d3d11_fragment_hlsl: {prefix}_D3D11_FRAGMENT_HLSL,"
        )
        .unwrap();
        writeln!(
            module,
            "        d3d11_vertex_dxbc: {prefix}_D3D11_VERTEX_DXBC,"
        )
        .unwrap();
        writeln!(
            module,
            "        d3d11_fragment_dxbc: {prefix}_D3D11_FRAGMENT_DXBC,"
        )
        .unwrap();
        writeln!(module, "        vertex_stride: {prefix}_VERTEX_STRIDE,").unwrap();
        writeln!(module, "        entry_points: {prefix}_ENTRY_POINTS,").unwrap();
        writeln!(
            module,
            "        vertex_attributes: {prefix}_VERTEX_ATTRIBUTES,"
        )
        .unwrap();
        writeln!(
            module,
            "        d3d11_resource_bindings: {prefix}_D3D11_RESOURCE_BINDINGS,"
        )
        .unwrap();
        writeln!(module, "    }},").unwrap();
    }
    module.push_str("];\n\n");

    module
        .push_str("pub(crate) fn core_shader(id: CoreShader) -> &'static CoreShaderArtifacts {\n");
    module.push_str("    match id {\n");
    for (index, artifact) in artifacts.iter().enumerate() {
        writeln!(
            module,
            "        CoreShader::{} => &CORE_SHADERS[{index}],",
            artifact.descriptor.variant_name
        )
        .unwrap();
    }
    module.push_str("    }\n");
    module.push_str("}\n");
    Ok(())
}
