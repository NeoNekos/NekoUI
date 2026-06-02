use std::collections::{BTreeMap, BTreeSet};

use naga::{
    AddressSpace, Binding, ImageClass, Scalar, ScalarKind, StorageAccess, TypeInner, VectorSize,
};

use super::{
    D3d11RegisterClass, D3d11ResourceMetadata, EntryPointMetadata, ShaderDescriptor,
    ShaderMetadata, ShaderStage, VertexAttributeMetadata, constant_name_from_snake_case,
    is_ascii_snake_case,
};

const ATTRIBUTE_SEMANTIC: &str = "LOC";

pub(crate) fn reflect_shader(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
) -> Result<ShaderMetadata, String> {
    let vertex_entry = only_entry_point(descriptor, module, naga::ShaderStage::Vertex)?;
    let fragment_entry = only_entry_point(descriptor, module, naga::ShaderStage::Fragment)?;
    let mut vertex_attributes = reflect_vertex_attributes(descriptor, module, vertex_entry)?;
    vertex_attributes.sort_by_key(|attribute| attribute.semantic_index);
    assign_packed_offsets(descriptor, &mut vertex_attributes)?;

    Ok(ShaderMetadata {
        entry_points: vec![
            EntryPointMetadata {
                stage: ShaderStage::Vertex,
                name: vertex_entry.name.clone(),
            },
            EntryPointMetadata {
                stage: ShaderStage::Fragment,
                name: fragment_entry.name.clone(),
            },
        ],
        vertex_stride: vertex_attributes
            .iter()
            .map(|attribute| attribute.byte_width)
            .sum(),
        vertex_attributes,
        d3d11_resources: reflect_d3d11_resources(descriptor, module)?,
    })
}

fn only_entry_point<'a>(
    descriptor: &ShaderDescriptor,
    module: &'a naga::Module,
    stage: naga::ShaderStage,
) -> Result<&'a naga::EntryPoint, String> {
    let matches = module
        .entry_points
        .iter()
        .filter(|entry_point| entry_point.stage == stage)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [entry_point] => Ok(*entry_point),
        [] => Err(format!(
            "{} must declare exactly one {:?} entry point; found none",
            descriptor.logical_id, stage
        )),
        _ => Err(format!(
            "{} must declare exactly one {:?} entry point; found {}",
            descriptor.logical_id,
            stage,
            matches.len()
        )),
    }
}

fn reflect_vertex_attributes(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
    vertex_entry: &naga::EntryPoint,
) -> Result<Vec<VertexAttributeMetadata>, String> {
    let mut attributes = Vec::new();
    for argument in &vertex_entry.function.arguments {
        if let Some(binding) = argument.binding.as_ref() {
            reflect_vertex_binding(
                descriptor,
                module,
                argument.name.as_deref(),
                argument.ty,
                binding,
                &mut attributes,
            )?;
            continue;
        }

        let TypeInner::Struct { members, .. } = &module.types[argument.ty].inner else {
            return Err(format!(
                "{} vertex argument `{}` has no binding and is not a struct",
                descriptor.logical_id,
                argument.name.as_deref().unwrap_or("<unnamed>")
            ));
        };
        for member in members {
            let Some(binding) = member.binding.as_ref() else {
                return Err(format!(
                    "{} vertex input struct member `{}` has no binding",
                    descriptor.logical_id,
                    member.name.as_deref().unwrap_or("<unnamed>")
                ));
            };
            reflect_vertex_binding(
                descriptor,
                module,
                member.name.as_deref(),
                member.ty,
                binding,
                &mut attributes,
            )?;
        }
    }

    if attributes.is_empty() {
        return Err(format!(
            "{} vertex entry point `{}` has no reflected vertex attributes",
            descriptor.logical_id, vertex_entry.name
        ));
    }

    Ok(attributes)
}

fn reflect_vertex_binding(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
    name: Option<&str>,
    ty: naga::Handle<naga::Type>,
    binding: &Binding,
    attributes: &mut Vec<VertexAttributeMetadata>,
) -> Result<(), String> {
    let Binding::Location { location, .. } = *binding else {
        return Ok(());
    };
    let name = name.ok_or_else(|| {
        format!(
            "{} vertex input at location {location} must have a WGSL member name",
            descriptor.logical_id
        )
    })?;
    if !is_ascii_snake_case(name) {
        return Err(format!(
            "{} vertex input `{name}` must be ASCII snake_case for generated constants",
            descriptor.logical_id
        ));
    }
    let (format, byte_width) = vertex_format(descriptor, module, ty, name)?;
    attributes.push(VertexAttributeMetadata {
        const_name: constant_name_from_snake_case(name),
        semantic: ATTRIBUTE_SEMANTIC,
        semantic_index: location,
        format,
        offset: 0,
        byte_width,
    });
    Ok(())
}

fn vertex_format(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    name: &str,
) -> Result<(&'static str, u32), String> {
    let float32 = Scalar {
        kind: ScalarKind::Float,
        width: 4,
    };
    match &module.types[ty].inner {
        TypeInner::Vector {
            size: VectorSize::Bi,
            scalar,
        } if *scalar == float32 => Ok(("R32G32_FLOAT", 8)),
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar,
        } if *scalar == float32 => Ok(("R32G32B32A32_FLOAT", 16)),
        ref other => Err(format!(
            "{} vertex input `{name}` uses unsupported v0 type {other:?}; expected vec2<f32> or vec4<f32>",
            descriptor.logical_id
        )),
    }
}

fn assign_packed_offsets(
    descriptor: &ShaderDescriptor,
    attributes: &mut [VertexAttributeMetadata],
) -> Result<(), String> {
    let mut seen_locations = BTreeSet::new();
    let mut seen_names = BTreeSet::new();
    let mut offset = 0_u32;
    for attribute in attributes {
        if !seen_locations.insert(attribute.semantic_index) {
            return Err(format!(
                "{} vertex location {} is declared more than once",
                descriptor.logical_id, attribute.semantic_index
            ));
        }
        if !seen_names.insert(attribute.const_name.clone()) {
            return Err(format!(
                "{} vertex constant {} is declared more than once",
                descriptor.logical_id, attribute.const_name
            ));
        }
        attribute.offset = offset;
        offset = offset
            .checked_add(attribute.byte_width)
            .ok_or_else(|| format!("{} vertex stride overflowed u32", descriptor.logical_id))?;
    }
    Ok(())
}

fn reflect_d3d11_resources(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
) -> Result<Vec<D3d11ResourceMetadata>, String> {
    let mut reflected = Vec::new();
    let mut seen_bindings = BTreeSet::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        if !seen_bindings.insert((binding.group, binding.binding)) {
            return Err(format!(
                "{} has duplicate resource binding @group({}) @binding({})",
                descriptor.logical_id, binding.group, binding.binding
            ));
        }
        let name = global.name.clone().ok_or_else(|| {
            format!(
                "{} resource @group({}) @binding({}) must have a WGSL name",
                descriptor.logical_id, binding.group, binding.binding
            )
        })?;
        if !is_ascii_snake_case(&name) {
            return Err(format!(
                "{} resource `{name}` must be ASCII snake_case for generated constants",
                descriptor.logical_id
            ));
        }
        reflected.push((
            binding.group,
            binding.binding,
            name.clone(),
            D3d11ResourceMetadata {
                register_class: classify_resource(descriptor, module, global)?,
                name,
                group: binding.group,
                binding: binding.binding,
                slot: 0,
            },
        ));
    }
    reflected
        .sort_by(|left, right| (&left.0, &left.1, &left.2).cmp(&(&right.0, &right.1, &right.2)));

    let mut next_slots = BTreeMap::from([
        (D3d11RegisterClass::Srv.label(), 0_u32),
        (D3d11RegisterClass::Sampler.label(), 0_u32),
        (D3d11RegisterClass::Cbv.label(), 0_u32),
        (D3d11RegisterClass::Uav.label(), 0_u32),
    ]);
    let mut resources = Vec::with_capacity(reflected.len());
    for (_, _, _, mut resource) in reflected {
        let label = resource.register_class.label();
        let next_slot = next_slots.get_mut(label).ok_or_else(|| {
            format!(
                "{} unknown D3D11 register class {label}",
                descriptor.logical_id
            )
        })?;
        resource.slot = *next_slot;
        *next_slot = next_slot.checked_add(1).ok_or_else(|| {
            format!(
                "{} D3D11 {} register allocation overflowed u32",
                descriptor.logical_id, label
            )
        })?;
        resources.push(resource);
    }
    Ok(resources)
}

fn classify_resource(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
    global: &naga::GlobalVariable,
) -> Result<D3d11RegisterClass, String> {
    match global.space {
        AddressSpace::Uniform => Ok(D3d11RegisterClass::Cbv),
        AddressSpace::Storage { access } => Ok(if writes_storage(access) {
            D3d11RegisterClass::Uav
        } else {
            D3d11RegisterClass::Srv
        }),
        AddressSpace::Handle => classify_handle_resource(descriptor, module, global.ty),
        ref other => Err(format!(
            "{} resource `{}` uses unsupported address space {other:?}",
            descriptor.logical_id,
            global.name.as_deref().unwrap_or("<unnamed>")
        )),
    }
}

fn classify_handle_resource(
    descriptor: &ShaderDescriptor,
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<D3d11RegisterClass, String> {
    match &module.types[ty].inner {
        TypeInner::Image { class, .. } => match class {
            ImageClass::Sampled { .. } | ImageClass::Depth { .. } | ImageClass::External => {
                Ok(D3d11RegisterClass::Srv)
            }
            ImageClass::Storage { access, .. } => Ok(if writes_storage(*access) {
                D3d11RegisterClass::Uav
            } else {
                D3d11RegisterClass::Srv
            }),
        },
        TypeInner::Sampler { .. } => Ok(D3d11RegisterClass::Sampler),
        TypeInner::BindingArray { base, .. } => classify_handle_resource(descriptor, module, *base),
        other => Err(format!(
            "{} handle resource uses unsupported type {other:?}",
            descriptor.logical_id
        )),
    }
}

fn writes_storage(access: StorageAccess) -> bool {
    access.intersects(StorageAccess::STORE | StorageAccess::ATOMIC)
}
