use crate::error::{NekoError, NekoResult};
use crate::render::{
    GLYPH_MONO_D3D11_FRAGMENT_DXBC, GLYPH_MONO_D3D11_VERTEX_DXBC, SOLID_RECT_D3D11_FRAGMENT_DXBC,
    SOLID_RECT_D3D11_VERTEX_DXBC,
};

const DXBC_MAGIC: &[u8] = b"DXBC";
const DXBC_PLACEHOLDER_PREFIX: &[u8] = b"NEKOUI_DXBC_PLACEHOLDER_V0";

pub(super) fn solid_rect_vertex_shader_bytes() -> NekoResult<&'static [u8]> {
    checked_bytes(SOLID_RECT_D3D11_VERTEX_DXBC, "solid_rect vertex DXBC")
}

pub(super) fn solid_rect_pixel_shader_bytes() -> NekoResult<&'static [u8]> {
    checked_bytes(SOLID_RECT_D3D11_FRAGMENT_DXBC, "solid_rect pixel DXBC")
}

pub(super) fn glyph_mono_vertex_shader_bytes() -> NekoResult<&'static [u8]> {
    checked_bytes(GLYPH_MONO_D3D11_VERTEX_DXBC, "glyph_mono vertex DXBC")
}

pub(super) fn glyph_mono_pixel_shader_bytes() -> NekoResult<&'static [u8]> {
    checked_bytes(GLYPH_MONO_D3D11_FRAGMENT_DXBC, "glyph_mono pixel DXBC")
}

fn checked_bytes(bytes: &'static [u8], label: &'static str) -> NekoResult<&'static [u8]> {
    if bytes.is_empty() {
        return Err(NekoError::unsupported(format!(
            "{label} is empty; build.rs must generate framework shader DXBC artifacts into OUT_DIR"
        )));
    }
    if bytes.starts_with(DXBC_PLACEHOLDER_PREFIX) {
        return Err(NekoError::unsupported(format!(
            "{label} is a placeholder; build.rs must fail instead of generating placeholder DXBC"
        )));
    }
    if !bytes.starts_with(DXBC_MAGIC) {
        return Err(NekoError::unsupported(format!(
            "{label} is not a DXBC shader blob; build.rs must generate valid DXBC artifacts into OUT_DIR"
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use crate::error::ErrorKind;

    use super::*;

    #[test]
    fn checked_bytes_rejects_empty_placeholder_and_non_dxbc_bytes() {
        assert_eq!(
            checked_bytes(b"DXBCreal", "test DXBC").unwrap(),
            b"DXBCreal"
        );
        assert_unsupported(checked_bytes(b"", "empty DXBC"));
        assert_unsupported(checked_bytes(
            b"NEKOUI_DXBC_PLACEHOLDER_V0",
            "placeholder DXBC",
        ));
        assert_unsupported(checked_bytes(b"not-dxbc", "invalid DXBC"));
    }

    fn assert_unsupported(result: NekoResult<&'static [u8]>) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
