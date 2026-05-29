use nekoui::prelude::*;

// Public integration coverage intentionally stays narrow: ordinary consumers must see
// the startup/prelude/facade surface, but deterministic runtime snapshot coverage uses
// crate-private harness tests in `src/app/tests.rs` so `run_test` and `TestRun` do not
// become ordinary public API.

#[derive(Debug)]
struct EmptyRoot;

impl Render for EmptyRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
    }
}

#[test]
fn application_run_is_the_public_startup_path_without_executing_native_loop() {
    fn accepts_startup(
        _application: Application,
        _startup: fn(&mut AppContext<'_>) -> NekoResult<()>,
    ) {
    }

    fn startup(cx: &mut AppContext<'_>) -> NekoResult<()> {
        cx.windows()
            .open(WindowOptions::new().title("Public Run"), |_| EmptyRoot)?;
        Ok(())
    }

    accepts_startup(Application::new(), startup);
}

#[test]
fn prelude_keeps_ordinary_app_surface_without_test_run_imports() {
    fn accepts_public_app_types(_application: Application, _result: fn() -> NekoResult<()>) {}

    fn result_factory() -> NekoResult<()> {
        Ok(())
    }

    accepts_public_app_types(Application::new(), result_factory);
}

#[test]
fn public_facades_do_not_leak_native_backend_types() {
    let facade_files = [
        "src/lib.rs",
        "src/prelude.rs",
        "src/app.rs",
        "src/diagnostic.rs",
        "src/element.rs",
        "src/interaction.rs",
        "src/layout.rs",
        "src/retained.rs",
        "src/scene.rs",
        "src/style.rs",
        "src/window.rs",
    ];
    let forbidden_tokens = [
        "windows::",
        "windows_sys::",
        "HWND",
        "HINSTANCE",
        "HMODULE",
        "ID3D11",
        "IDXGI",
        "D3D11",
        "DXGI",
        "raw_window_handle",
        "RawWindowHandle",
        "HasWindowHandle",
        "Win32WindowHandle",
        "cosmic_text",
        "fontdb::",
        "swash::",
        "etagere::",
        "TextLayoutHandle",
        "TextLayoutRef",
        "GlyphRun",
        "GlyphKey",
        "GlyphBitmap",
        "GlyphDemand",
        "GlyphInstance",
        "GlyphAtlas",
        "PreparedFrame",
        "DrawItemKind",
        "UploadIntent",
        "UploadPlan",
        "D3d11GlyphMonoPipeline",
    ];

    let mut leaks = Vec::new();
    for file in facade_files {
        let source = std::fs::read_to_string(file).unwrap();
        for token in forbidden_tokens {
            if source.contains(token) {
                leaks.push(format!("{file} contains {token}"));
            }
        }
    }

    assert_eq!(leaks, Vec::<String>::new());
}
