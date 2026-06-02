use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use super::{ShaderDescriptor, constant_name_from_snake_case, is_ascii_snake_case};

pub(crate) fn discover_shaders(source_dir: &Path) -> Result<Vec<ShaderDescriptor>, String> {
    let mut source_files = Vec::new();
    let entries = fs::read_dir(source_dir).map_err(|error| {
        format!(
            "failed to read framework shader source directory {}: {error}",
            source_dir.display()
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read framework shader directory entry in {}: {error}",
                source_dir.display()
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("wesl") {
            source_files.push(path);
        }
    }
    source_files.sort();

    if source_files.is_empty() {
        return Err(format!(
            "no framework shader WESL files found in {}",
            source_dir.display()
        ));
    }

    let mut seen_ids = BTreeSet::new();
    let mut descriptors = Vec::with_capacity(source_files.len());
    for path in source_files {
        let source_file = path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .ok_or_else(|| {
                format!(
                    "framework shader path is not valid UTF-8: {}",
                    path.display()
                )
            })?
            .to_owned();
        let file_stem = path
            .file_stem()
            .and_then(|file_stem| file_stem.to_str())
            .ok_or_else(|| {
                format!(
                    "framework shader path is not valid UTF-8: {}",
                    path.display()
                )
            })?
            .to_owned();
        if !is_ascii_snake_case(&file_stem) {
            return Err(format!(
                "framework shader filename `{source_file}` must be ASCII snake_case"
            ));
        }

        let logical_id = format!("core.{file_stem}");
        if !seen_ids.insert(logical_id.clone()) {
            return Err(format!("duplicate framework shader id `{logical_id}`"));
        }

        descriptors.push(ShaderDescriptor {
            logical_id,
            variant_name: variant_name_from_snake_case(&file_stem),
            constant_prefix: constant_name_from_snake_case(&file_stem),
            wesl_package_path: format!("package::{file_stem}"),
            file_stem,
            source_file,
        });
    }

    Ok(descriptors)
}

fn variant_name_from_snake_case(name: &str) -> String {
    let mut variant = String::new();
    for part in name.split('_') {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            variant.push(first.to_ascii_uppercase());
            variant.extend(chars);
        }
    }
    variant
}
