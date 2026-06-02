use naga::back::hlsl as naga_hlsl;

use super::{
    D3d11RegisterClass, D3d11ResourceMetadata, EntryPointMetadata, ShaderDescriptor,
    ShaderMetadata, ShaderStage,
};

pub(crate) struct HlslStageOutput {
    pub(crate) source: String,
    pub(crate) entry_point: String,
}

struct HlslWriteRequest<'a> {
    descriptor: &'a ShaderDescriptor,
    stage: ShaderStage,
    wgsl_entry_point: &'a EntryPointMetadata,
    module: &'a naga::Module,
    info: &'a naga::valid::ModuleInfo,
    options: &'a naga_hlsl::Options,
    pipeline_options: &'a naga_hlsl::PipelineOptions,
    fragment_entry_point: Option<&'a naga_hlsl::FragmentEntryPoint<'a>>,
}

pub(crate) fn generate_hlsl_sm5(
    descriptor: &ShaderDescriptor,
    metadata: &ShaderMetadata,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: ShaderStage,
) -> Result<HlslStageOutput, String> {
    let options = naga_hlsl::Options {
        shader_model: naga_hlsl::ShaderModel::V5_0,
        binding_map: binding_map(&metadata.d3d11_resources),
        sampler_buffer_binding_map: sampler_buffer_binding_map(&metadata.d3d11_resources)?,
        fake_missing_bindings: false,
        ..naga_hlsl::Options::default()
    };
    let wgsl_entry_point = metadata.entry_point(stage)?;
    let pipeline_options = naga_hlsl::PipelineOptions {
        entry_point: Some((stage.naga_stage(), wgsl_entry_point.name.clone())),
    };
    let fragment_entry_point = match stage {
        ShaderStage::Vertex => {
            let fragment_entry = metadata.entry_point(ShaderStage::Fragment)?;
            Some(
                naga_hlsl::FragmentEntryPoint::new(module, &fragment_entry.name).ok_or_else(
                    || {
                        format!(
                            "{} reflected fragment entry point `{}` was not found for HLSL vertex interface trimming",
                            descriptor.logical_id, fragment_entry.name
                        )
                    },
                )?,
            )
        }
        ShaderStage::Fragment => None,
    };

    let (mut source, reflection) = write_hlsl_source(HlslWriteRequest {
        descriptor,
        stage,
        wgsl_entry_point,
        module,
        info,
        options: &options,
        pipeline_options: &pipeline_options,
        fragment_entry_point: fragment_entry_point.as_ref(),
    })?;
    let entry_point = select_hlsl_entry_point(descriptor, stage, wgsl_entry_point, reflection)?;
    rewrite_d3d11_sm5_sampler_bindings(descriptor, &mut source, &metadata.d3d11_resources)?;
    if !contains_entry_point(&source, &entry_point) {
        return Err(format!(
            "{} HLSL reflection selected {} entry point `{entry_point}`, but the generated source does not contain that function",
            descriptor.logical_id,
            stage.label()
        ));
    }
    Ok(HlslStageOutput {
        source,
        entry_point,
    })
}

fn write_hlsl_source(
    request: HlslWriteRequest<'_>,
) -> Result<(String, naga_hlsl::ReflectionInfo), String> {
    let mut source = String::new();
    let mut writer = naga_hlsl::Writer::new(&mut source, request.options, request.pipeline_options);
    let reflection = writer
        .write(request.module, request.info, request.fragment_entry_point)
        .map_err(|error| {
            format!(
                "{} HLSL generation failed for {} entry point `{}`: {error}",
                request.descriptor.logical_id,
                request.stage.label(),
                request.wgsl_entry_point.name
            )
        })?;
    drop(writer);
    Ok((source, reflection))
}

fn binding_map(resources: &[D3d11ResourceMetadata]) -> naga_hlsl::BindingMap {
    let mut binding_map = naga_hlsl::BindingMap::default();
    for resource in resources {
        binding_map.insert(
            naga::ResourceBinding {
                group: resource.group,
                binding: resource.binding,
            },
            naga_hlsl::BindTarget {
                space: 0,
                register: resource.slot,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
    }
    binding_map
}

fn sampler_buffer_binding_map(
    resources: &[D3d11ResourceMetadata],
) -> Result<naga_hlsl::SamplerIndexBufferBindingMap, String> {
    let mut groups = resources
        .iter()
        .filter(|resource| resource.register_class == D3d11RegisterClass::Sampler)
        .map(|resource| resource.group)
        .collect::<Vec<_>>();
    groups.sort_unstable();
    groups.dedup();

    let next_srv_register = resources
        .iter()
        .filter(|resource| resource.register_class == D3d11RegisterClass::Srv)
        .map(|resource| resource.slot)
        .max()
        .map_or(0, |slot| slot.saturating_add(1));

    let mut map = naga_hlsl::SamplerIndexBufferBindingMap::default();
    for (index, group) in groups.into_iter().enumerate() {
        let register = next_srv_register
            .checked_add(u32::try_from(index).map_err(|_| {
                "D3D11 sampler index buffer register allocation overflowed u32".to_owned()
            })?)
            .ok_or_else(|| {
                "D3D11 sampler index buffer register allocation overflowed u32".to_owned()
            })?;
        map.insert(
            naga_hlsl::SamplerIndexBufferKey { group },
            naga_hlsl::BindTarget {
                space: 0,
                register,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
    }
    Ok(map)
}

fn rewrite_d3d11_sm5_sampler_bindings(
    descriptor: &ShaderDescriptor,
    source: &mut String,
    resources: &[D3d11ResourceMetadata],
) -> Result<(), String> {
    let sampler_resources = resources
        .iter()
        .filter(|resource| resource.register_class == D3d11RegisterClass::Sampler)
        .collect::<Vec<_>>();
    if sampler_resources.is_empty() {
        return Ok(());
    }

    let sampler_index_buffers = sampler_index_buffers(resources)?;

    let mut rewritten = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("static const SamplerComparisonState ") {
            return Err(format!(
                "{} generated comparison sampler alias HLSL, which is not supported by the D3D11 SM5 direct sampler rewrite",
                descriptor.logical_id
            ));
        }
        if trimmed.starts_with("SamplerComparisonState nagaComparisonSamplerHeap[2048]:") {
            continue;
        }
        if trimmed.starts_with("SamplerState nagaSamplerHeap[2048]:") {
            continue;
        }
        if sampler_index_buffers
            .iter()
            .any(|buffer| buffer.matches_declaration(trimmed))
        {
            continue;
        }

        if let Some(replacement) =
            direct_sampler_declaration(descriptor, trimmed, &sampler_resources)?
        {
            rewritten.push_str(line_indent(line));
            rewritten.push_str(&replacement);
            rewritten.push('\n');
            continue;
        }

        rewritten.push_str(line);
        rewritten.push('\n');
    }
    *source = rewritten;
    validate_d3d11_sm5_sampler_rewrite(descriptor, source, &sampler_resources)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SamplerIndexBuffer {
    group: u32,
    register: u32,
}

impl SamplerIndexBuffer {
    fn matches_declaration(self, trimmed_line: &str) -> bool {
        let Some(rest) = trimmed_line.strip_prefix("StructuredBuffer<uint> ") else {
            return false;
        };
        let Some((name, register)) = rest.split_once(':') else {
            return false;
        };
        name.trim() == format!("nagaGroup{}SamplerIndexArray", self.group)
            && register.trim() == format!("register(t{}, space0);", self.register)
    }
}

fn direct_sampler_declaration(
    descriptor: &ShaderDescriptor,
    trimmed_line: &str,
    resources: &[&D3d11ResourceMetadata],
) -> Result<Option<String>, String> {
    const PREFIX: &str = "static const SamplerState ";
    let Some(remainder) = trimmed_line.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let Some((name, expression)) = remainder.split_once(" = ") else {
        return Ok(None);
    };
    if !expression.starts_with("nagaSamplerHeap[") {
        return Ok(None);
    }
    let Some(resource) = resources
        .iter()
        .copied()
        .find(|resource| resource.name == name)
    else {
        return Err(format!(
            "{} generated sampler heap declaration for unknown sampler `{name}`",
            descriptor.logical_id
        ));
    };
    let expected = format!(
        "nagaSamplerHeap[nagaGroup{}SamplerIndexArray[{}]];",
        resource.group, resource.slot
    );
    if expression != expected {
        return Err(format!(
            "{} generated unsupported D3D11 sampler heap expression for `{name}`: `{expression}`; expected `{expected}`",
            descriptor.logical_id
        ));
    }
    Ok(Some(format!(
        "SamplerState {name}: register(s{});",
        resource.slot
    )))
}

fn validate_d3d11_sm5_sampler_rewrite(
    descriptor: &ShaderDescriptor,
    source: &str,
    sampler_resources: &[&D3d11ResourceMetadata],
) -> Result<(), String> {
    for forbidden in [
        "nagaSamplerHeap",
        "nagaComparisonSamplerHeap",
        "SamplerIndexArray",
        "space0",
    ] {
        if source.contains(forbidden) {
            let line = source
                .lines()
                .find(|line| line.contains(forbidden))
                .unwrap_or("<missing line>");
            return Err(format!(
                "{} D3D11 SM5 sampler rewrite left unsupported HLSL token `{forbidden}` in generated source line `{}`",
                descriptor.logical_id,
                line.trim()
            ));
        }
    }
    for resource in sampler_resources {
        let declaration = format!(
            "SamplerState {}: register(s{});",
            resource.name, resource.slot
        );
        if !source.contains(&declaration) {
            return Err(format!(
                "{} D3D11 SM5 sampler rewrite did not emit direct sampler declaration `{declaration}`",
                descriptor.logical_id
            ));
        }
    }
    Ok(())
}

fn line_indent(line: &str) -> &str {
    let indent_len = line
        .char_indices()
        .find_map(|(index, character)| (!character.is_whitespace()).then_some(index))
        .unwrap_or(line.len());
    &line[..indent_len]
}

fn select_hlsl_entry_point(
    descriptor: &ShaderDescriptor,
    stage: ShaderStage,
    wgsl_entry_point: &EntryPointMetadata,
    reflection: naga_hlsl::ReflectionInfo,
) -> Result<String, String> {
    match reflection.entry_point_names.as_slice() {
        [Ok(entry_point)] => Ok(entry_point.clone()),
        [Err(error)] => Err(format!(
            "{} HLSL reflection failed for {} entry point `{}`: {error}",
            descriptor.logical_id,
            stage.label(),
            wgsl_entry_point.name
        )),
        [] => Err(format!(
            "{} HLSL reflection returned no {} entry point for `{}`",
            descriptor.logical_id,
            stage.label(),
            wgsl_entry_point.name
        )),
        entries => Err(format!(
            "{} HLSL reflection returned {} {} entry points for `{}`; expected exactly one",
            descriptor.logical_id,
            entries.len(),
            stage.label(),
            wgsl_entry_point.name
        )),
    }
}

fn contains_entry_point(source: &str, entry_point: &str) -> bool {
    let needle = format!("{entry_point}(");
    source.match_indices(&needle).any(|(index, _)| {
        source[..index]
            .chars()
            .next_back()
            .map(|character| !is_identifier_character(character))
            .unwrap_or(true)
    })
}

fn is_identifier_character(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}

fn sampler_index_buffers(
    resources: &[D3d11ResourceMetadata],
) -> Result<Vec<SamplerIndexBuffer>, String> {
    let mut groups = resources
        .iter()
        .filter(|resource| resource.register_class == D3d11RegisterClass::Sampler)
        .map(|resource| resource.group)
        .collect::<Vec<_>>();
    groups.sort_unstable();
    groups.dedup();

    let next_srv_register = resources
        .iter()
        .filter(|resource| resource.register_class == D3d11RegisterClass::Srv)
        .map(|resource| resource.slot)
        .max()
        .map_or(0, |slot| slot.saturating_add(1));

    groups
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            let register = next_srv_register
                .checked_add(u32::try_from(index).map_err(|_| {
                    "D3D11 sampler index buffer register allocation overflowed u32".to_owned()
                })?)
                .ok_or_else(|| {
                    "D3D11 sampler index buffer register allocation overflowed u32".to_owned()
                })?;
            Ok(SamplerIndexBuffer { group, register })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> ShaderDescriptor {
        ShaderDescriptor {
            logical_id: "core.glyph_mono".to_owned(),
            file_stem: "glyph_mono".to_owned(),
            variant_name: "GlyphMono".to_owned(),
            constant_prefix: "GLYPH_MONO".to_owned(),
            source_file: "glyph_mono.wesl".to_owned(),
            wesl_package_path: "package::glyph_mono".to_owned(),
        }
    }

    fn glyph_resources() -> Vec<D3d11ResourceMetadata> {
        vec![
            D3d11ResourceMetadata {
                name: "glyph_atlas".to_owned(),
                group: 0,
                binding: 0,
                register_class: D3d11RegisterClass::Srv,
                slot: 0,
            },
            D3d11ResourceMetadata {
                name: "glyph_sampler".to_owned(),
                group: 0,
                binding: 1,
                register_class: D3d11RegisterClass::Sampler,
                slot: 0,
            },
        ]
    }

    #[test]
    fn d3d11_sampler_rewrite_replaces_naga_heap_with_direct_sampler() {
        let mut source = concat!(
            "SamplerState nagaSamplerHeap[2048]: register(s0, space0);\n",
            "StructuredBuffer<uint> nagaGroup0SamplerIndexArray: register(t1, space0);\n",
            "static const SamplerState glyph_sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[0]];\n",
            "float4 fs_main() : SV_Target0 { return float4(1.0, 1.0, 1.0, 1.0); }\n",
        )
        .to_owned();

        rewrite_d3d11_sm5_sampler_bindings(&descriptor(), &mut source, &glyph_resources()).unwrap();

        assert!(source.contains("SamplerState glyph_sampler: register(s0);"));
        assert!(!source.contains("nagaSamplerHeap"));
        assert!(!source.contains("SamplerIndexArray"));
        assert!(!source.contains("space0"));
    }

    #[test]
    fn d3d11_sampler_rewrite_rejects_unexpected_sampler_expression() {
        let mut source = concat!(
            "SamplerState nagaSamplerHeap[2048]: register(s0, space0);\n",
            "StructuredBuffer<uint> nagaGroup0SamplerIndexArray: register(t1, space0);\n",
            "static const SamplerState glyph_sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[1]];\n",
        )
        .to_owned();

        let error =
            rewrite_d3d11_sm5_sampler_bindings(&descriptor(), &mut source, &glyph_resources())
                .unwrap_err();

        assert!(error.contains("unsupported D3D11 sampler heap expression"));
    }

    #[test]
    fn d3d11_sampler_rewrite_removes_unused_comparison_heap() {
        let mut source = concat!(
            "SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space0);\n",
            "SamplerState nagaSamplerHeap[2048]: register(s0, space0);\n",
            "StructuredBuffer<uint> nagaGroup0SamplerIndexArray: register(t1, space0);\n",
            "static const SamplerState glyph_sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[0]];\n",
            "float4 fs_main() : SV_Target0 { return float4(1.0, 1.0, 1.0, 1.0); }\n",
        )
        .to_owned();

        rewrite_d3d11_sm5_sampler_bindings(&descriptor(), &mut source, &glyph_resources()).unwrap();

        assert!(!source.contains("nagaComparisonSamplerHeap"));
        assert!(source.contains("SamplerState glyph_sampler: register(s0);"));
    }

    #[test]
    fn d3d11_sampler_rewrite_rejects_comparison_sampler_alias() {
        let mut source = concat!(
            "SamplerState nagaSamplerHeap[2048]: register(s0, space0);\n",
            "StructuredBuffer<uint> nagaGroup0SamplerIndexArray: register(t1, space0);\n",
            "static const SamplerComparisonState glyph_sampler = nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[0]];\n",
        )
        .to_owned();

        let error =
            rewrite_d3d11_sm5_sampler_bindings(&descriptor(), &mut source, &glyph_resources())
                .unwrap_err();

        assert!(error.contains("comparison sampler alias"));
    }
}
