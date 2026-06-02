use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{ShaderDescriptor, ShaderStage};

const DXBC_MAGIC: &[u8] = b"DXBC";
const DXBC_HELP_STR: &str = "Install the Windows SDK (https://developer.microsoft.com/windows/downloads/windows-sdk/) for fxc.exe, then add the compiler directory to PATH or set NEKOUI_FXC to the full fxc.exe path. DirectX Shader Compiler releases are documented at https://github.com/microsoft/DirectXShaderCompiler/releases, but DXC/DXIL is not accepted as a replacement for the current D3D11 SM5 DXBC artifacts.";

#[derive(Debug)]
pub(crate) struct Compiler {
    path: PathBuf,
}

pub(crate) fn compile_dxbc(
    compiler: &Compiler,
    descriptor: &ShaderDescriptor,
    hlsl_path: &Path,
    entry_point: &str,
    stage: ShaderStage,
    output_path: &Path,
) -> Result<(), String> {
    let mut command = Command::new(&compiler.path);
    command
        .arg("/nologo")
        .arg("/T")
        .arg(stage.hlsl_profile())
        .arg("/E")
        .arg(entry_point)
        .arg("/Fo")
        .arg(output_path)
        .arg(hlsl_path);

    let output = command.output().map_err(|error| {
        format!(
            "{} failed to launch DXBC compiler {}: {error}\n{}",
            descriptor.logical_id,
            compiler.path.to_string_lossy(),
            DXBC_HELP_STR
        )
    })?;

    if !output.status.success() {
        return Err(format!(
            "{} DXBC compile failed for reflected HLSL entry point {} {} with status {}\nstdout:\n{}\nstderr:\n{}\n{}",
            descriptor.logical_id,
            entry_point,
            stage.hlsl_profile(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
            DXBC_HELP_STR
        ));
    }

    let bytes = fs::read(output_path).map_err(|error| {
        format!(
            "{} DXBC compiler did not write {}: {error}",
            descriptor.logical_id,
            output_path.display()
        )
    })?;
    if !bytes.starts_with(DXBC_MAGIC) {
        return Err(format!(
            "{} DXBC compiler output {} is not a DXBC blob; no placeholder artifact will be generated",
            descriptor.logical_id,
            output_path.display()
        ));
    }
    Ok(())
}

pub(crate) fn find_dxbc_compiler() -> Result<Compiler, String> {
    if let Some(path) = env::var_os("NEKOUI_FXC") {
        return Ok(Compiler {
            path: PathBuf::from(path),
        });
    }
    if command_is_available("fxc") {
        return Ok(Compiler {
            path: PathBuf::from("fxc"),
        });
    }

    Err(format!(
        "missing D3D11 DXBC compiler: fxc was not found in PATH and NEKOUI_FXC is not set.\n{}",
        DXBC_HELP_STR
    ))
}

fn command_is_available(program: &str) -> bool {
    Command::new(program).arg("--help").output().is_ok()
        || Command::new(program).arg("/?").output().is_ok()
}
