use crate::app::{Context, Render};
use crate::element::{Element, IntoElement, div, text};
use crate::layout::LayoutSize;
use crate::platform::{PhysicalSize, PlatformFact, Renderability};
use crate::render::{DrawItemKind, RenderPass, prepare_frame_graph};
use crate::runtime::Runtime;
use crate::scene::ResourceDemandKind;
use crate::style::{Color, StyleExt, opacity};
use crate::window::WindowOptions;
use nekoui_shader_types::ShaderBackendTarget;

#[derive(Debug)]
struct TestRoot {
    root: Element,
}

impl TestRoot {
    fn new(root: impl IntoElement) -> Self {
        Self {
            root: root.into_element(),
        }
    }
}

impl Render for TestRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.root.clone()
    }
}

fn compile_scene(root: impl IntoElement) -> crate::scene::PaintScene {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(root))
        .unwrap();
    runtime.scene_snapshot(window).unwrap()
}

#[test]
fn prepared_frame_preserves_scene_order_in_draw_items() {
    let scene = compile_scene(
        div()
            .key("root")
            .bg(Color::rgb(1, 2, 3))
            .child(text("A").key("a"))
            .child(text("B").key("b")),
    );
    let frame = prepare_frame_graph(&scene);
    let scene_orders = scene
        .fragments()
        .iter()
        .map(|fragment| fragment.order())
        .collect::<Vec<_>>();
    let draw_orders = frame
        .draw_items()
        .iter()
        .map(|item| item.order())
        .collect::<Vec<_>>();

    assert_eq!(draw_orders, scene_orders);
    assert_eq!(frame.stats().draw_item_count, scene.fragments().len());
}

#[test]
fn upload_plan_is_explicit_and_precedes_dependent_draw_pass() {
    let scene = compile_scene(div().key("root").child(text("Glyphs").key("label")));
    let frame = prepare_frame_graph(&scene);
    let glyph_upload = frame
        .upload_plan()
        .intents()
        .iter()
        .find(|intent| intent.kind() == ResourceDemandKind::Glyph)
        .unwrap();
    let text_order = frame
        .draw_items()
        .iter()
        .find(|item| matches!(item.kind(), DrawItemKind::Text { .. }))
        .unwrap()
        .order();

    assert_eq!(frame.passes()[0].class(), RenderPass::Upload);
    assert_eq!(
        frame.passes()[0].upload_count(),
        frame.upload_plan().intents().len()
    );
    assert_eq!(frame.passes()[1].class(), RenderPass::MainColor);
    assert_eq!(
        frame.passes()[1].draw_orders(),
        frame
            .draw_items()
            .iter()
            .map(|item| item.order())
            .collect::<Vec<_>>()
    );
    assert!(glyph_upload.dependent_draw_orders().contains(&text_order));
    assert!(glyph_upload.glyphs().is_some());
    assert_eq!(
        glyph_upload.owner_node_id(),
        frame.draw_items()[0].node_id()
    );
    assert_eq!(
        glyph_upload.expected_generation(),
        scene.generation().text_generation()
    );
}

#[test]
fn text_draw_items_carry_private_layout_and_color() {
    let scene = compile_scene(text("Glyphs").text_color(Color::rgb(11, 22, 33)));
    let frame = prepare_frame_graph(&scene);
    let text = frame
        .draw_items()
        .iter()
        .find_map(|item| match item.kind() {
            DrawItemKind::Text { layout, color, .. } => Some((layout, *color)),
            _ => None,
        })
        .unwrap();

    assert_eq!(text.1, Color::rgb(11, 22, 33));
    assert!(!text.0.glyphs().is_empty());
    assert!(text.0.glyph_demands().len() <= text.0.glyphs().len());
}

#[test]
fn unsupported_fragments_are_carried_as_draw_items_and_upload_intent() {
    let scene = compile_scene(div().child(text("layer").opacity(opacity(0.5))));
    let frame = prepare_frame_graph(&scene);

    assert!(frame.draw_items().iter().any(|item| matches!(
        item.kind(),
        DrawItemKind::Unsupported {
            capability: "partial_opacity_layer"
        }
    )));
    assert!(
        frame
            .upload_plan()
            .intents()
            .iter()
            .any(|intent| intent.kind() == ResourceDemandKind::Unsupported)
    );
    assert_eq!(frame.stats().unsupported_fragment_count, 1);
}

#[test]
fn runtime_stores_prepared_frame_after_scene_publication() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().child(text("prepared")))
        })
        .unwrap();
    let scene = runtime.scene_snapshot(window).unwrap();
    let prepared = runtime
        .state()
        .prepared_frame_snapshot(window.id())
        .unwrap();
    let record = runtime.state().window(window.into()).unwrap();
    let surface_generation = record.surface_generation();

    assert_eq!(prepared.generation().scene(), &scene.generation());
    assert_eq!(
        prepared.generation().surface_generation(),
        Some(surface_generation)
    );
    assert!(prepared.is_current_for_scene(&scene));
    assert!(prepared.is_current_for_scene_and_surface(&scene, surface_generation));
    assert_eq!(
        prepared.context().logical_viewport_size(),
        record.viewport().logical_size()
    );
    assert_eq!(
        prepared.context().physical_surface_size(),
        record.physical_size()
    );
    assert_eq!(
        prepared.context().scale_factor(),
        record.viewport().scale_factor()
    );
    assert_eq!(
        prepared.context().viewport_generation(),
        record.viewport().generation()
    );
    assert_eq!(
        prepared.context().surface_generation(),
        Some(surface_generation)
    );
    assert_eq!(runtime.performance_report().render.frame_graph_count, 1);
    assert_eq!(runtime.performance_report().render.prepared_frame_count, 1);
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("render.frame_graph"),
        1
    );
}

#[test]
fn prepared_frame_generation_rejects_stale_scene_snapshot() {
    let first_scene = compile_scene(div().key("root").child(text("first").key("label")));
    let second_scene = compile_scene(div().key("root").child(text("second").key("label")));
    let first_frame = prepare_frame_graph(&first_scene);

    assert!(first_frame.is_current_for_scene(&first_scene));
    assert!(!first_frame.is_current_for_scene(&second_scene));
}

#[test]
fn draw_items_retain_fragment_bounds_for_backend_prepare() {
    let scene = compile_scene(div().key("root").bg(Color::rgb(1, 2, 3)));
    let frame = prepare_frame_graph(&scene);
    let fragment = scene.fragments().first().unwrap();
    let item = frame.draw_items().first().unwrap();

    assert_eq!(item.rect(), fragment.rect());
}

#[test]
fn rect_draw_items_preserve_color_and_order() {
    let scene = compile_scene(
        div()
            .key("root")
            .bg(Color::rgb(1, 2, 3))
            .child(div().key("first").bg(Color::rgba(4, 5, 6, 7)))
            .child(div().key("second").bg(Color::rgb(8, 9, 10))),
    );
    let frame = prepare_frame_graph(&scene);
    let rects = frame
        .draw_items()
        .iter()
        .filter_map(|item| match item.kind() {
            DrawItemKind::Rect { color } => Some((item.order(), *color)),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(rects.len(), 3);
    assert_eq!(rects[0].1, Color::rgb(1, 2, 3));
    assert_eq!(rects[1].1, Color::rgba(4, 5, 6, 7));
    assert_eq!(rects[2].1, Color::rgb(8, 9, 10));
    assert!(rects[0].0.raw() < rects[1].0.raw());
    assert!(rects[1].0.raw() < rects[2].0.raw());
}

#[test]
fn shader_manifest_and_snapshots_are_consistent_for_solid_rect() {
    assert_eq!(
        crate::render::SOLID_RECT_MANIFEST.shader.as_str(),
        "core.solid_rect"
    );
    assert_eq!(crate::render::SOLID_RECT_MANIFEST.vertex_stride, 24);
    assert_eq!(
        crate::render::SOLID_RECT_MANIFEST.vertex_attributes[0].offset,
        0
    );
    assert_eq!(
        crate::render::SOLID_RECT_MANIFEST.vertex_attributes[1].offset,
        8
    );
    assert!(crate::render::SOLID_RECT_WGSL.contains("fn vs_main"));
    assert!(crate::render::SOLID_RECT_D3D11_VERTEX_HLSL.contains("vs_main"));
    assert!(crate::render::SOLID_RECT_D3D11_FRAGMENT_HLSL.contains("fs_main"));

    assert_artifact(
        ShaderBackendTarget::Wgsl,
        crate::render::SOLID_RECT_WGSL.as_bytes(),
        false,
    );
    assert_artifact(
        ShaderBackendTarget::D3d11Sm5VertexHlsl,
        crate::render::SOLID_RECT_D3D11_VERTEX_HLSL.as_bytes(),
        false,
    );
    assert_artifact(
        ShaderBackendTarget::D3d11Sm5FragmentHlsl,
        crate::render::SOLID_RECT_D3D11_FRAGMENT_HLSL.as_bytes(),
        false,
    );
    assert_artifact(
        ShaderBackendTarget::D3d11Sm5VertexDxbc,
        crate::render::SOLID_RECT_D3D11_VERTEX_DXBC,
        true,
    );
    assert_artifact(
        ShaderBackendTarget::D3d11Sm5FragmentDxbc,
        crate::render::SOLID_RECT_D3D11_FRAGMENT_DXBC,
        true,
    );

    let manifest = std::fs::read_to_string("shaders/generated/solid_rect.manifest.toml").unwrap();
    for entry_point in crate::render::SOLID_RECT_MANIFEST.entry_points {
        assert_manifest_entry_point(&manifest, entry_point);
    }
    for target in crate::render::SOLID_RECT_MANIFEST.targets {
        assert_manifest_target(&manifest, target);
    }
}

#[test]
fn shader_manifest_and_snapshots_are_consistent_for_glyph_mono() {
    assert_eq!(
        crate::render::GLYPH_MONO_MANIFEST.shader.as_str(),
        "core.glyph_mono"
    );
    assert_eq!(crate::render::GLYPH_MONO_MANIFEST.vertex_stride, 32);
    assert_eq!(
        crate::render::GLYPH_MONO_MANIFEST.vertex_attributes[0].offset,
        0
    );
    assert_eq!(
        crate::render::GLYPH_MONO_MANIFEST.vertex_attributes[1].offset,
        8
    );
    assert_eq!(
        crate::render::GLYPH_MONO_MANIFEST.vertex_attributes[2].offset,
        16
    );
    assert!(crate::render::GLYPH_MONO_WGSL.contains("textureSample"));
    assert!(crate::render::GLYPH_MONO_D3D11_VERTEX_HLSL.contains("TEXCOORD0"));
    assert!(crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("glyph_atlas"));
    assert_ne!(
        crate::render::GLYPH_MONO_WGSL,
        crate::render::SOLID_RECT_WGSL
    );
    assert_ne!(
        crate::render::GLYPH_MONO_D3D11_VERTEX_DXBC,
        crate::render::SOLID_RECT_D3D11_VERTEX_DXBC
    );
    assert_ne!(
        crate::render::GLYPH_MONO_D3D11_FRAGMENT_DXBC,
        crate::render::SOLID_RECT_D3D11_FRAGMENT_DXBC
    );

    for (target, bytes, checked_binary) in [
        (
            ShaderBackendTarget::Wgsl,
            crate::render::GLYPH_MONO_WGSL.as_bytes(),
            false,
        ),
        (
            ShaderBackendTarget::D3d11Sm5VertexHlsl,
            crate::render::GLYPH_MONO_D3D11_VERTEX_HLSL.as_bytes(),
            false,
        ),
        (
            ShaderBackendTarget::D3d11Sm5FragmentHlsl,
            crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.as_bytes(),
            false,
        ),
        (
            ShaderBackendTarget::D3d11Sm5VertexDxbc,
            crate::render::GLYPH_MONO_D3D11_VERTEX_DXBC,
            true,
        ),
        (
            ShaderBackendTarget::D3d11Sm5FragmentDxbc,
            crate::render::GLYPH_MONO_D3D11_FRAGMENT_DXBC,
            true,
        ),
    ] {
        assert_artifact_for_manifest(
            &crate::render::GLYPH_MONO_MANIFEST,
            target,
            bytes,
            checked_binary,
        );
    }
    let manifest = std::fs::read_to_string("shaders/generated/glyph_mono.manifest.toml").unwrap();
    for entry_point in crate::render::GLYPH_MONO_MANIFEST.entry_points {
        assert_manifest_entry_point(&manifest, entry_point);
    }
    for target in crate::render::GLYPH_MONO_MANIFEST.targets {
        assert_manifest_target(&manifest, target);
    }
}

fn assert_artifact(target: ShaderBackendTarget, bytes: &[u8], checked_binary: bool) {
    assert_artifact_for_manifest(
        &crate::render::SOLID_RECT_MANIFEST,
        target,
        bytes,
        checked_binary,
    );
}

fn assert_artifact_for_manifest(
    manifest: &nekoui_shader_types::CoreShaderManifest,
    target: ShaderBackendTarget,
    bytes: &[u8],
    checked_binary: bool,
) {
    let manifest = manifest
        .targets
        .iter()
        .find(|candidate| candidate.target == target)
        .unwrap();
    assert_eq!(manifest.checked_binary, checked_binary);
    assert_eq!(sha256_hex(bytes), manifest.sha256);
    if checked_binary {
        assert!(bytes.starts_with(b"DXBC"));
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn assert_manifest_target(manifest: &str, target: &nekoui_shader_types::ShaderArtifactTarget) {
    let expected_target = format!("target = \"{}\"", target.target.as_str());
    let block = manifest
        .split("[[targets]]")
        .skip(1)
        .find(|block| block.contains(&expected_target))
        .unwrap_or_else(|| {
            panic!(
                "missing manifest target block for {}",
                target.target.as_str()
            )
        });

    assert!(block.contains(&format!("path = \"{}\"", target.path)));
    assert!(block.contains(&format!("sha256 = \"{}\"", target.sha256)));
    assert!(block.contains(&format!("checked_binary = {}", target.checked_binary)));
}

fn assert_manifest_entry_point(
    manifest: &str,
    entry_point: &nekoui_shader_types::ShaderEntryPoint,
) {
    let expected_stage = format!("stage = \"{}\"", entry_point.stage.as_str());
    let block = manifest
        .split("[[entry_points]]")
        .skip(1)
        .find(|block| block.contains(&expected_stage))
        .unwrap_or_else(|| {
            panic!(
                "missing manifest entry point block for {}",
                entry_point.stage.as_str()
            )
        });

    assert!(block.contains(&format!("name = \"{}\"", entry_point.name)));
}

#[test]
fn runtime_dependencies_do_not_include_wesl_or_naga_tooling() {
    let manifest = std::fs::read_to_string("Cargo.toml").unwrap();
    let dependencies = manifest
        .split("[dependencies]")
        .nth(1)
        .unwrap()
        .split("[target.")
        .next()
        .unwrap();

    assert!(!dependencies.contains("naga"));
    assert!(!dependencies.contains("wesl"));
}

#[test]
fn not_renderable_redraw_does_not_prepare_pointless_frames_until_restore() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(
            WindowOptions::new().logical_size(LayoutSize::new(100.0, 100.0)),
            |_| TestRoot::new(div().child(text("stable"))),
        )
        .unwrap();
    let initial_count = runtime.performance_report().render.frame_graph_count;

    runtime
        .ingest_platform_fact(PlatformFact::PhysicalSizeChanged {
            handle: window.into(),
            physical_size: PhysicalSize::ZERO,
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::RedrawRequested {
            handle: window.into(),
        })
        .unwrap();

    assert_eq!(
        runtime.performance_report().render.frame_graph_count,
        initial_count
    );
    assert_eq!(
        runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .renderability(),
        Renderability::ZeroSize
    );

    runtime
        .ingest_platform_fact(PlatformFact::PhysicalSizeChanged {
            handle: window.into(),
            physical_size: PhysicalSize::new(100, 100),
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    assert_eq!(
        runtime.performance_report().render.frame_graph_count,
        initial_count + 1
    );
}

#[test]
fn diagnostics_and_performance_project_render_frame_graph() {
    let scene = compile_scene(div().child(text("A")).child(text("B")));
    let frame = prepare_frame_graph(&scene);
    let mut runtime = Runtime::new();
    runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().child(text("A")))
        })
        .unwrap();
    let report = runtime.performance_report();
    let diagnostics = runtime.diagnostics().snapshot();
    let render_record = diagnostics
        .records()
        .iter()
        .find(|record| record.operation == "render.frame_graph")
        .unwrap();

    assert!(frame.stats().pass_count >= 1);
    assert_eq!(report.render.frame_graph_count, 1);
    assert!(report.render.last_frame_graph.surface_generation.is_some());
    assert_eq!(
        report.render.last_frame_graph.pass_count as u64,
        report.render.pass_count
    );
    assert!(report.phase_durations.contains_key("render.frame_graph"));
    assert_ne!(
        render_record.fields.get("surface_generation").unwrap(),
        "none"
    );
}

fn stress_scene(item_count: usize, text_every: usize) -> crate::scene::PaintScene {
    let mut root = div().key("root").bg(Color::rgb(1, 2, 3));
    for index in 0..item_count {
        if index % text_every == 0 {
            root = root.child(text(format!("item {index}")).key(format!("text-{index}")));
        } else {
            root = root.child(div().key(format!("rect-{index}")).bg(Color::rgb(4, 5, 6)));
        }
    }
    compile_scene(root)
}

fn run_manual_frame_graph_stress(item_count: usize, text_every: usize) {
    let scene = stress_scene(item_count, text_every);
    let frame = prepare_frame_graph(&scene);

    assert_eq!(frame.draw_items().len(), scene.fragments().len());
    assert_eq!(
        frame.stats().upload_intent_count,
        scene.resource_demands().len()
    );
}

#[test]
#[ignore = "manual render frame graph stress: 10k simple"]
fn manual_render_frame_graph_stress_10k_simple() {
    run_manual_frame_graph_stress(10_000, usize::MAX);
}

#[test]
#[ignore = "manual render frame graph stress: 10k text-heavy"]
fn manual_render_frame_graph_stress_10k_text_heavy() {
    run_manual_frame_graph_stress(10_000, 1);
}

#[test]
#[ignore = "manual render frame graph stress: 10k mixed"]
fn manual_render_frame_graph_stress_10k_mixed() {
    run_manual_frame_graph_stress(10_000, 3);
}

#[test]
#[ignore = "manual render frame graph stress: 100k simple"]
fn manual_render_frame_graph_stress_100k_simple() {
    run_manual_frame_graph_stress(100_000, usize::MAX);
}

#[test]
#[ignore = "manual render frame graph stress: 100k text-heavy"]
fn manual_render_frame_graph_stress_100k_text_heavy() {
    run_manual_frame_graph_stress(100_000, 1);
}

#[test]
#[ignore = "manual render frame graph stress: 100k mixed"]
fn manual_render_frame_graph_stress_100k_mixed() {
    run_manual_frame_graph_stress(100_000, 3);
}

#[test]
#[ignore = "manual render frame graph stress: 1m simple"]
fn manual_render_frame_graph_stress_1m_simple() {
    run_manual_frame_graph_stress(1_000_000, usize::MAX);
}

#[test]
#[ignore = "manual render frame graph stress: 1m text-heavy"]
fn manual_render_frame_graph_stress_1m_text_heavy() {
    run_manual_frame_graph_stress(1_000_000, 1);
}

#[test]
#[ignore = "manual render frame graph stress: 1m mixed"]
fn manual_render_frame_graph_stress_1m_mixed() {
    run_manual_frame_graph_stress(1_000_000, 3);
}
