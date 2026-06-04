use crate::app::{Context, Render};
use crate::element::{Element, IntoElement, div, input, text};
use crate::layout::{LayoutRect, LayoutSize};
use crate::platform::{PhysicalSize, PlatformFact, Renderability};
use crate::render::{DrawItemKind, RenderPass, prepare_frame_graph};
use crate::runtime::Runtime;
use crate::scene::ResourceDemandKind;
use crate::style::{Color, Overflow, StyleExt, opacity, px};
use crate::window::WindowOptions;

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
fn clip_push_payload_and_order_survive_scene_to_render() {
    let scene = compile_scene(
        div().key("root").child(
            div()
                .key("clipper")
                .w(px(100.0))
                .h(px(80.0))
                .overflow(Overflow::Scroll)
                .child(text("inside").key("label").h(px(160.0))),
        ),
    );
    let frame = prepare_frame_graph(&scene);
    let scene_clips = scene
        .fragments()
        .iter()
        .filter_map(|fragment| match fragment.kind() {
            crate::scene::PaintFragmentKind::ClipPush { clip } => Some((fragment.order(), *clip)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let draw_clips = frame
        .draw_items()
        .iter()
        .filter_map(|item| match item.kind() {
            DrawItemKind::ClipPush { clip } => Some((item.order(), *clip)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let clip_push_order = draw_clips[0].0;
    let text_order = frame
        .draw_items()
        .iter()
        .find(|item| matches!(item.kind(), DrawItemKind::Text { .. }))
        .unwrap()
        .order();
    let clip_pop_order = frame
        .draw_items()
        .iter()
        .find(|item| matches!(item.kind(), DrawItemKind::ClipPop))
        .unwrap()
        .order();

    assert_eq!(draw_clips, scene_clips);
    assert_eq!(draw_clips[0].1, LayoutRect::new(0.0, 0.0, 100.0, 80.0));
    assert!(clip_push_order.raw() < text_order.raw());
    assert!(text_order.raw() < clip_pop_order.raw());
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
fn input_text_draw_items_carry_private_content_clip() {
    let scene = compile_scene(input("AAAA AAAA AAAA").w(px(36.0)));
    let frame = prepare_frame_graph(&scene);
    let text = frame
        .draw_items()
        .iter()
        .find_map(|item| match item.kind() {
            DrawItemKind::Text { clip, .. } => Some((*clip, item.rect())),
            _ => None,
        })
        .unwrap();

    assert!(text.0.is_some());
    assert_eq!(text.0.unwrap().width(), 36.0);
    assert!(text.1.x() <= text.0.unwrap().x());
}

#[test]
fn ordinary_text_draw_items_carry_private_content_clip() {
    let scene = compile_scene(text("AAAA AAAA AAAA").w(px(36.0)));
    let frame = prepare_frame_graph(&scene);
    let text = frame
        .draw_items()
        .iter()
        .find_map(|item| match item.kind() {
            DrawItemKind::Text { clip, .. } => Some(*clip),
            _ => None,
        })
        .unwrap();

    assert!(text.is_some());
    assert_eq!(text.unwrap().width(), 36.0);
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
fn box_shape_draw_items_preserve_fill_and_order() {
    let scene = compile_scene(
        div()
            .key("root")
            .bg(Color::rgb(1, 2, 3))
            .child(div().key("first").bg(Color::rgba(4, 5, 6, 7)))
            .child(div().key("second").bg(Color::rgb(8, 9, 10))),
    );
    let frame = prepare_frame_graph(&scene);
    let shapes = frame
        .draw_items()
        .iter()
        .filter_map(|item| match item.kind() {
            DrawItemKind::BoxShape { shape } => Some((item.order(), shape.fill().unwrap())),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(shapes.len(), 3);
    assert_eq!(shapes[0].1, Color::rgb(1, 2, 3));
    assert_eq!(shapes[1].1, Color::rgba(4, 5, 6, 7));
    assert_eq!(shapes[2].1, Color::rgb(8, 9, 10));
    assert!(shapes[0].0.raw() < shapes[1].0.raw());
    assert!(shapes[1].0.raw() < shapes[2].0.raw());
}

#[test]
fn box_shape_draw_items_preserve_opacity_and_stats_count_prepared_shapes() {
    let scene = compile_scene(
        div()
            .key("root")
            .child(
                div()
                    .key("transparent")
                    .bg(Color::rgb(1, 2, 3))
                    .opacity(opacity(0.25)),
            )
            .child(div().key("wide-color").bg(Color::oklch(0.5, 0.1, 120.0))),
    );
    let frame = prepare_frame_graph(&scene);
    let shapes = frame
        .draw_items()
        .iter()
        .filter_map(|item| match item.kind() {
            DrawItemKind::BoxShape { shape } => Some(*shape),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(shapes.len(), 2);
    assert_eq!(shapes[0].opacity(), opacity(0.25));
    assert_eq!(shapes[1].fill(), Some(Color::oklch(0.5, 0.1, 120.0)));
    assert_eq!(frame.stats().box_shape_count, shapes.len());
}

#[test]
fn generated_framework_shader_registry_contains_required_artifacts() {
    assert_eq!(crate::render::CORE_SHADERS.len(), 3);
    assert_generated_shader(
        crate::render::core_shader(crate::render::CoreShader::BoxShape),
        "core.box_shape",
        96,
        &[
            ("LOC", 0, 0, "R32G32_FLOAT"),
            ("LOC", 1, 8, "R32G32_FLOAT"),
            ("LOC", 2, 16, "R32G32B32A32_FLOAT"),
            ("LOC", 3, 32, "R32G32B32A32_FLOAT"),
            ("LOC", 4, 48, "R32G32B32A32_FLOAT"),
            ("LOC", 5, 64, "R32G32B32A32_FLOAT"),
            ("LOC", 6, 80, "R32G32B32A32_FLOAT"),
        ],
    );
    assert_generated_shader(
        crate::render::core_shader(crate::render::CoreShader::GlyphMono),
        "core.glyph_mono",
        32,
        &[
            ("LOC", 0, 0, "R32G32_FLOAT"),
            ("LOC", 1, 8, "R32G32_FLOAT"),
            ("LOC", 2, 16, "R32G32B32A32_FLOAT"),
        ],
    );
    assert_generated_shader(
        crate::render::core_shader(crate::render::CoreShader::GlyphColor),
        "core.glyph_color",
        32,
        &[
            ("LOC", 0, 0, "R32G32_FLOAT"),
            ("LOC", 1, 8, "R32G32_FLOAT"),
            ("LOC", 2, 16, "R32G32B32A32_FLOAT"),
        ],
    );
}

#[test]
fn generated_framework_shader_artifacts_are_complete_and_distinct() {
    assert!(crate::render::BOX_SHAPE_WGSL.contains("fn vs_main"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("vs_main"));
    assert!(crate::render::BOX_SHAPE_D3D11_FRAGMENT_HLSL.contains("fs_main"));
    assert_target_dxbc_bytes(crate::render::BOX_SHAPE_D3D11_VERTEX_DXBC);
    assert_target_dxbc_bytes(crate::render::BOX_SHAPE_D3D11_FRAGMENT_DXBC);

    assert!(crate::render::GLYPH_MONO_WGSL.contains("textureSample"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC0"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC1"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC2"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC3"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC4"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC5"));
    assert!(crate::render::BOX_SHAPE_D3D11_VERTEX_HLSL.contains("LOC6"));
    assert!(crate::render::GLYPH_MONO_D3D11_VERTEX_HLSL.contains("LOC0"));
    assert!(crate::render::GLYPH_MONO_D3D11_VERTEX_HLSL.contains("LOC1"));
    assert!(crate::render::GLYPH_MONO_D3D11_VERTEX_HLSL.contains("LOC2"));
    assert!(crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("glyph_atlas"));
    assert!(crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("fs_main("));
    assert!(crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("SamplerState glyph_sampler"));
    assert!(!crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("nagaSamplerHeap"));
    assert!(!crate::render::GLYPH_MONO_D3D11_FRAGMENT_HLSL.contains("nagaGroup0SamplerIndexArray"));
    assert_target_dxbc_bytes(crate::render::GLYPH_MONO_D3D11_VERTEX_DXBC);
    assert_target_dxbc_bytes(crate::render::GLYPH_MONO_D3D11_FRAGMENT_DXBC);

    assert!(crate::render::GLYPH_COLOR_WGSL.contains("textureSample"));
    assert!(crate::render::GLYPH_COLOR_WGSL.contains("sampled.rgb"));
    assert!(!crate::render::GLYPH_COLOR_WGSL.contains("input.color.rgb"));
    assert!(crate::render::GLYPH_COLOR_D3D11_VERTEX_HLSL.contains("LOC0"));
    assert!(crate::render::GLYPH_COLOR_D3D11_VERTEX_HLSL.contains("LOC1"));
    assert!(crate::render::GLYPH_COLOR_D3D11_VERTEX_HLSL.contains("LOC2"));
    assert!(crate::render::GLYPH_COLOR_D3D11_FRAGMENT_HLSL.contains("glyph_atlas"));
    assert!(crate::render::GLYPH_COLOR_D3D11_FRAGMENT_HLSL.contains("fs_main("));
    assert!(crate::render::GLYPH_COLOR_D3D11_FRAGMENT_HLSL.contains("SamplerState glyph_sampler"));
    assert_target_dxbc_bytes(crate::render::GLYPH_COLOR_D3D11_VERTEX_DXBC);
    assert_target_dxbc_bytes(crate::render::GLYPH_COLOR_D3D11_FRAGMENT_DXBC);

    assert_ne!(
        crate::render::GLYPH_MONO_WGSL,
        crate::render::BOX_SHAPE_WGSL
    );
    assert_ne!(
        crate::render::GLYPH_COLOR_WGSL,
        crate::render::GLYPH_MONO_WGSL
    );
    if cfg!(target_os = "windows") {
        assert_ne!(
            crate::render::GLYPH_MONO_D3D11_VERTEX_DXBC,
            crate::render::BOX_SHAPE_D3D11_VERTEX_DXBC
        );
        assert_ne!(
            crate::render::GLYPH_MONO_D3D11_FRAGMENT_DXBC,
            crate::render::BOX_SHAPE_D3D11_FRAGMENT_DXBC
        );
        assert_ne!(
            crate::render::GLYPH_COLOR_D3D11_FRAGMENT_DXBC,
            crate::render::GLYPH_MONO_D3D11_FRAGMENT_DXBC
        );
    }
}

#[test]
fn box_shape_shader_source_matches_straight_alpha_blend_contract() {
    let source = std::fs::read_to_string("src/platform/shader/box_shape.wesl").unwrap();

    assert_box_shape_shader_uses_straight_alpha(&source);
    assert_box_shape_shader_uses_straight_alpha(crate::render::BOX_SHAPE_WGSL);
    assert_box_shape_shader_carries_per_edge_and_corner_payload(&source);
    assert_box_shape_shader_carries_per_edge_and_corner_payload(crate::render::BOX_SHAPE_WGSL);
}

fn assert_box_shape_shader_uses_straight_alpha(source: &str) {
    assert!(source.contains("border_alpha > 0.0"));
    assert!(source.contains("output_rgb"));
    assert!(source.contains("return vec4<f32>(output_rgb, output_alpha)"));
    assert!(!source.contains("let fill = input.fill_color * fill_alpha"));
    assert!(!source.contains("return fill + border"));
}

fn assert_box_shape_shader_carries_per_edge_and_corner_payload(source: &str) {
    assert!(source.contains("corner_radii: vec4<f32>"));
    assert!(source.contains("border_widths: vec4<f32>"));
    assert!(source.contains("inner_corner_radii(corner_radii, border_widths)"));
}

#[test]
fn framework_shader_source_and_generated_artifacts_use_approved_locations() {
    assert!(std::path::Path::new("src/platform/shader/box_shape.wesl").is_file());
    assert!(std::path::Path::new("src/platform/shader/glyph_mono.wesl").is_file());
    assert!(std::path::Path::new("src/platform/shader/glyph_color.wesl").is_file());
    assert!(!std::path::Path::new("shaders").exists());
    assert!(!std::path::Path::new("../nekoui-shader-types").exists());
    assert!(!std::path::Path::new("../nekoui-shader-build").exists());
}

#[test]
fn build_script_dxbc_fail_fast_diagnostic_is_actionable() {
    let build_script = build_script_sources();

    assert!(build_script.contains("NEKOUI_FXC"));
    assert!(build_script.contains("PATH"));
    assert!(
        build_script.contains("https://developer.microsoft.com/windows/downloads/windows-sdk/")
    );
    assert!(build_script.contains("https://github.com/microsoft/DirectXShaderCompiler/releases"));
    assert!(build_script.contains("no placeholder artifact will be generated"));
    assert!(!build_script.contains("NEKOUI_DXBC_PLACEHOLDER_V0"));
}

#[test]
fn framework_shader_build_script_does_not_reuse_checked_in_artifacts() {
    let render_shaders = std::fs::read_to_string("src/render/shaders.rs").unwrap();
    let build_script = build_script_sources();

    assert!(render_shaders.contains("OUT_DIR"));
    for source in [render_shaders.as_str(), build_script.as_str()] {
        assert!(!source.contains("shaders/generated"));
        assert!(!source.contains("shaders/artifacts"));
        assert!(!source.contains("shaders/framework"));
        assert!(!source.contains("nekoui_shader_types"));
    }
}

#[test]
fn generated_framework_shader_metadata_reflects_glyph_resources() {
    assert_glyph_resource_bindings(
        crate::render::core_shader(crate::render::CoreShader::GlyphMono),
        crate::render::GLYPH_MONO_GLYPH_ATLAS_D3D11_SRV_SLOT,
        crate::render::GLYPH_MONO_GLYPH_SAMPLER_D3D11_SAMPLER_SLOT,
    );
    assert_glyph_resource_bindings(
        crate::render::core_shader(crate::render::CoreShader::GlyphColor),
        crate::render::GLYPH_COLOR_GLYPH_ATLAS_D3D11_SRV_SLOT,
        crate::render::GLYPH_COLOR_GLYPH_SAMPLER_D3D11_SAMPLER_SLOT,
    );
}

fn assert_glyph_resource_bindings(
    shader: &crate::render::CoreShaderArtifacts,
    atlas_slot: u32,
    sampler_slot: u32,
) {
    assert_eq!(shader.d3d11_resource_bindings.len(), 2);
    assert_eq!(
        shader.d3d11_resource_bindings[0],
        crate::render::D3d11ResourceBinding {
            name: "glyph_atlas",
            group: 0,
            binding: 0,
            register_class: "srv",
            slot: atlas_slot,
        }
    );
    assert_eq!(
        shader.d3d11_resource_bindings[1],
        crate::render::D3d11ResourceBinding {
            name: "glyph_sampler",
            group: 0,
            binding: 1,
            register_class: "sampler",
            slot: sampler_slot,
        }
    );
    assert_eq!(atlas_slot, 0);
    assert_eq!(sampler_slot, 0);
}

fn build_script_sources() -> String {
    let mut combined = std::fs::read_to_string("build.rs").unwrap();
    for path in [
        "../nekoui-build/src/lib.rs",
        "../nekoui-build/src/shader/mod.rs",
        "../nekoui-build/src/shader/discovery.rs",
        "../nekoui-build/src/shader/dxbc.rs",
        "../nekoui-build/src/shader/hlsl.rs",
        "../nekoui-build/src/shader/metadata.rs",
        "../nekoui-build/src/shader/rust_module.rs",
    ] {
        combined.push_str(&std::fs::read_to_string(path).unwrap());
    }
    combined
}

fn assert_generated_shader(
    shader: &crate::render::CoreShaderArtifacts,
    name: &str,
    vertex_stride: u32,
    attributes: &[(&str, u32, u32, &str)],
) {
    assert_eq!(shader.name, name);
    assert_eq!(shader.vertex_stride, vertex_stride);
    assert_eq!(shader.entry_points.len(), 2);
    assert_eq!(shader.entry_points[0].stage, "vertex");
    assert_eq!(shader.entry_points[0].name, "vs_main");
    assert_eq!(shader.entry_points[1].stage, "fragment");
    assert_eq!(shader.entry_points[1].name, "fs_main");
    assert_eq!(shader.vertex_attributes.len(), attributes.len());
    assert!(!shader.wgsl.is_empty());
    assert!(!shader.d3d11_vertex_hlsl.is_empty());
    assert!(!shader.d3d11_fragment_hlsl.is_empty());
    assert_target_dxbc_bytes(shader.d3d11_vertex_dxbc);
    assert_target_dxbc_bytes(shader.d3d11_fragment_dxbc);

    for (actual, expected) in shader.vertex_attributes.iter().zip(attributes) {
        assert_eq!(actual.semantic, expected.0);
        assert_eq!(actual.semantic_index, expected.1);
        assert_eq!(actual.offset, expected.2);
        assert_eq!(actual.format, expected.3);
    }
}

fn assert_target_dxbc_bytes(bytes: &[u8]) {
    if cfg!(target_os = "windows") {
        assert!(bytes.starts_with(b"DXBC"));
    } else {
        assert!(bytes.is_empty());
    }
}

#[test]
fn runtime_dependencies_do_not_include_wesl_or_naga_tooling() {
    let manifest = std::fs::read_to_string("Cargo.toml").unwrap();
    let dependencies = manifest
        .split("[dependencies]")
        .nth(1)
        .unwrap()
        .split("[build-dependencies]")
        .next()
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
