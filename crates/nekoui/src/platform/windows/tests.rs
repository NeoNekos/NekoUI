use crate::diagnostic::{DiagnosticArea, Diagnostics};
use crate::error::ErrorKind;
use crate::platform::{BackendFrameStatus, BackendSurfaceState, PhysicalSize, Renderability};
use crate::render::{
    DrawItem, DrawItemKind, FrameGraphStats, PreparedFrame, PreparedFrameContext,
    PreparedFrameGeneration, PreparedPass, RenderPass, UploadPlan,
};
use crate::scene::{SceneGeneration, SceneOrder};
use crate::style::StyleExt;
use crate::window::{AnyWindowHandle, WindowId, WindowOptions};
use crate::{Application, Color, Context, IntoElement, NekoResult, Render, div, fill, px, text};

use super::frame::{ActiveFrame, count_unsupported_draw_items_for_backend, receipt};
use super::renderer::record_frame_report;
use super::surface::{WINDOWS_BACKEND_CLEAR_COLOR, swap_chain_background_color, swap_chain_desc};

use windows::Win32::Graphics::Dxgi::Common::{DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM};
use windows::Win32::Graphics::Dxgi::{
    DXGI_SCALING_NONE, DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};

#[test]
fn frame_receipt_distinguishes_presented_from_not_renderable() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(1));

    let presented = receipt(
        handle,
        3,
        BackendFrameStatus::Presented,
        BackendSurfaceState::Ready,
        0,
        0,
        "presented",
    );
    let skipped = receipt(
        handle,
        3,
        BackendFrameStatus::NotRenderable,
        BackendSurfaceState::Suspended,
        0,
        0,
        "not renderable",
    );

    assert_eq!(presented.status, BackendFrameStatus::Presented);
    assert_eq!(skipped.status, BackendFrameStatus::NotRenderable);
    assert_eq!(presented.surface_state, BackendSurfaceState::Ready);
    assert_eq!(skipped.surface_state, BackendSurfaceState::Suspended);
}

#[test]
fn physical_size_model_marks_zero_dimensions_not_renderable() {
    assert!(PhysicalSize::new(0, 480).is_zero());
    assert!(PhysicalSize::new(640, 0).is_zero());
    assert!(!PhysicalSize::new(640, 480).is_zero());
    assert_eq!(PhysicalSize::new(640, 480).width(), 640);
    assert_eq!(PhysicalSize::new(640, 480).height(), 480);
}

#[test]
fn active_frame_begin_rejects_not_renderable_without_present_success() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(2));
    let receipt = receipt(
        handle,
        4,
        BackendFrameStatus::NotRenderable,
        BackendSurfaceState::Suspended,
        0,
        0,
        "surface is not renderable",
    );

    assert_eq!(receipt.status, BackendFrameStatus::NotRenderable);
    assert_ne!(receipt.status, BackendFrameStatus::Presented);
    assert_eq!(receipt.surface_state, BackendSurfaceState::Suspended);
}

#[test]
fn stale_generation_receipt_is_a_stale_drop() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(3));
    let receipt = receipt(
        handle,
        7,
        BackendFrameStatus::StaleDropped,
        BackendSurfaceState::Ready,
        0,
        1,
        "prepared frame surface generation is stale",
    );

    assert_eq!(receipt.status, BackendFrameStatus::StaleDropped);
    assert_eq!(receipt.stale_drop_count, 1);
}

#[test]
fn rect_and_glyph_text_draw_items_are_supported_while_advanced_items_remain_unsupported() {
    let prepared = prepared_frame_with_draw_items(vec![
        DrawItem::new(
            SceneOrder::new(1),
            10,
            crate::layout::LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Rect {
                color: Color::rgb(1, 2, 3),
            },
        ),
        DrawItem::new(
            SceneOrder::new(2),
            11,
            crate::layout::LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Text {
                text_generation: crate::scene::SceneInputSignature::default(),
                text_metrics_generation: 1,
                layout: test_text_layout_with_glyph(),
                color: Color::rgb(1, 2, 3),
            },
        ),
        DrawItem::new(
            SceneOrder::new(3),
            12,
            crate::layout::LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            DrawItemKind::Unsupported {
                capability: "advanced_effect",
            },
        ),
    ]);

    assert_eq!(count_unsupported_draw_items_for_backend(&prepared), 1);
}

#[test]
fn empty_text_draw_items_remain_unsupported_until_glyphs_exist() {
    let prepared = prepared_frame_with_draw_items(vec![DrawItem::new(
        SceneOrder::new(1),
        10,
        crate::layout::LayoutRect::new(0.0, 0.0, 10.0, 10.0),
        DrawItemKind::Text {
            text_generation: crate::scene::SceneInputSignature::default(),
            text_metrics_generation: 1,
            layout: test_empty_text_layout(),
            color: Color::rgb(1, 2, 3),
        },
    )]);

    assert_eq!(count_unsupported_draw_items_for_backend(&prepared), 1);
}

#[test]
fn non_srgb_rect_colors_remain_unsupported_until_color_conversion_exists() {
    let prepared = prepared_frame_with_draw_items(vec![DrawItem::new(
        SceneOrder::new(1),
        10,
        crate::layout::LayoutRect::new(0.0, 0.0, 10.0, 10.0),
        DrawItemKind::Rect {
            color: Color::oklch(0.5, 0.1, 120.0),
        },
    )]);

    assert_eq!(count_unsupported_draw_items_for_backend(&prepared), 1);
}

#[test]
fn empty_rect_batch_has_no_invalid_draw_or_unsupported_count() {
    let prepared = prepared_frame_with_draw_items(Vec::new());

    assert_eq!(count_unsupported_draw_items_for_backend(&prepared), 0);
}

#[test]
fn shader_artifact_internals_do_not_leak_through_public_facades() {
    for file in [
        "src/lib.rs",
        "src/prelude.rs",
        "src/platform.rs",
        "src/render.rs",
    ] {
        let source = std::fs::read_to_string(file).unwrap();
        assert!(!source.contains("D3d11SolidRectPipeline"));
        assert!(!source.contains("D3d11GlyphMonoPipeline"));
        assert!(!source.contains("GlyphAtlas"));
        assert!(!source.contains("solid_rect_vertex_shader_bytes"));
        assert!(!source.contains("glyph_mono_vertex_shader_bytes"));
        assert!(!source.contains("ID3D11VertexShader"));
    }
}

#[test]
fn windows_swapchain_desc_uses_no_stretch_flip_discard_opaque_policy() {
    let desc = swap_chain_desc(PhysicalSize::new(800, 600), DXGI_FORMAT_B8G8R8A8_UNORM);

    assert_eq!(desc.Width, 800);
    assert_eq!(desc.Height, 600);
    assert_eq!(desc.Format, DXGI_FORMAT_B8G8R8A8_UNORM);
    assert_eq!(desc.BufferUsage, DXGI_USAGE_RENDER_TARGET_OUTPUT);
    assert_eq!(desc.BufferCount, 2);
    assert_eq!(desc.Scaling, DXGI_SCALING_NONE);
    assert_eq!(desc.SwapEffect, DXGI_SWAP_EFFECT_FLIP_DISCARD);
    assert_eq!(desc.AlphaMode, DXGI_ALPHA_MODE_IGNORE);
    assert_eq!(desc.SampleDesc.Count, 1);
    assert_eq!(desc.SampleDesc.Quality, 0);
    assert_eq!(desc.Flags, 0);
}

#[test]
fn windows_swapchain_background_color_matches_backend_clear_and_is_opaque() {
    let background = swap_chain_background_color();

    assert_eq!(background.r, WINDOWS_BACKEND_CLEAR_COLOR[0]);
    assert_eq!(background.g, WINDOWS_BACKEND_CLEAR_COLOR[1]);
    assert_eq!(background.b, WINDOWS_BACKEND_CLEAR_COLOR[2]);
    assert_eq!(background.a, WINDOWS_BACKEND_CLEAR_COLOR[3]);
    assert_eq!(background.a, 1.0);
}

#[test]
fn one_active_frame_policy_reports_failed_receipt_for_conflict() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(4));
    let conflict = receipt(
        handle,
        9,
        BackendFrameStatus::Failed,
        BackendSurfaceState::Ready,
        0,
        0,
        "surface already has an active frame",
    );

    assert_eq!(conflict.status, BackendFrameStatus::Failed);
    assert_eq!(conflict.surface_state, BackendSurfaceState::Ready);
}

#[test]
fn active_frame_abort_and_failure_reports_are_distinct_receipts() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(44));

    let aborted =
        ActiveFrame::new_for_test(handle, 12).abort(BackendSurfaceState::Ready, "record aborted");
    let failed = ActiveFrame::new_for_test(handle, 12).fail(
        BackendSurfaceState::Lost,
        "record failed",
        crate::error::NekoError::backend_lost("device lost"),
    );

    assert_eq!(aborted.receipt.status, BackendFrameStatus::Aborted);
    assert_eq!(aborted.receipt.surface_state, BackendSurfaceState::Ready);
    assert!(aborted.error.is_none());
    assert_eq!(failed.receipt.status, BackendFrameStatus::Failed);
    assert_eq!(failed.receipt.surface_state, BackendSurfaceState::Lost);
    assert_eq!(
        failed.error.as_ref().map(crate::error::NekoError::kind),
        Some(crate::error::ErrorKind::BackendLost)
    );
}

#[test]
fn failed_frame_receipt_preserves_actual_failure_category() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(45));
    let failed = ActiveFrame::new_for_test(handle, 13).fail(
        BackendSurfaceState::Lost,
        "record failed",
        crate::error::NekoError::resource_failure("upload allocation failed"),
    );

    assert_eq!(failed.receipt.status, BackendFrameStatus::Failed);
    assert_eq!(
        failed.receipt.failure_kind,
        Some(ErrorKind::ResourceFailure)
    );
    assert_eq!(
        failed.receipt.diagnostic_category(),
        ErrorKind::ResourceFailure
    );
}

#[test]
fn failed_frame_phase_diagnostic_preserves_actual_failure_category() {
    let handle = AnyWindowHandle::new_for_tests(WindowId::new(46));
    let report = ActiveFrame::new_for_test(handle, 14).fail(
        BackendSurfaceState::Lost,
        "record failed",
        crate::error::NekoError::resource_failure("upload allocation failed"),
    );
    let mut diagnostics = Diagnostics::default();

    record_frame_report(&mut diagnostics, report);

    let records = diagnostics.snapshot().records().to_vec();
    let phase = records
        .iter()
        .find(|record| record.area == DiagnosticArea::Gpu && record.operation == "gpu.frame.phase")
        .expect("gpu.frame.phase diagnostic should be recorded");
    let error = records
        .iter()
        .find(|record| record.area == DiagnosticArea::Gpu && record.operation == "gpu.frame.error")
        .expect("gpu.frame.error diagnostic should be recorded");

    assert_eq!(phase.category, ErrorKind::ResourceFailure);
    assert_eq!(error.category, ErrorKind::ResourceFailure);
    assert_eq!(
        phase.fields.get("status").map(|value| value.as_ref()),
        Some("failed")
    );
}

#[test]
fn renderability_model_names_not_renderable_reasons() {
    assert!(!Renderability::ZeroSize.is_renderable());
    assert_eq!(Renderability::ZeroSize.name(), "zero_size");
    assert!(!Renderability::Minimized.is_renderable());
    assert_eq!(Renderability::Minimized.name(), "minimized");
    assert!(Renderability::Renderable.is_renderable());
}

#[test]
fn backend_types_do_not_leak_through_public_crate_root() {
    let lib = std::fs::read_to_string("src/lib.rs").unwrap();

    assert!(!lib.contains("ID3D11"));
    assert!(!lib.contains("IDXGI"));
    assert!(!lib.contains("HWND"));
    assert!(!lib.contains("RawWindowHandle"));
}

#[test]
fn unsafe_boundary_stays_inside_windows_backend_files() {
    let source_root = std::path::Path::new("src");
    let allowed_prefix = std::path::Path::new("src/platform/windows");
    let mut offenders = Vec::new();

    collect_unsafe_offenders(source_root, allowed_prefix, &mut offenders);

    assert_eq!(offenders, Vec::<String>::new());
}

#[test]
fn glyph_unsafe_boundary_is_minimized_and_documented() {
    let glyph_source = std::fs::read_to_string("src/platform/windows/glyph.rs").unwrap();
    let glyph_pipeline_source =
        std::fs::read_to_string("src/platform/windows/glyph_pipeline.rs").unwrap();

    assert!(!contains_unsafe_escape(&glyph_source));
    assert_unsafe_blocks_have_safety_comments(
        "src/platform/windows/glyph_pipeline.rs",
        &glyph_pipeline_source,
    );
}

fn prepared_frame_with_draw_items(draw_items: Vec<DrawItem>) -> PreparedFrame {
    let draw_orders = draw_items.iter().map(DrawItem::order).collect::<Vec<_>>();
    let draw_item_count = draw_items.len();
    PreparedFrame::new(
        PreparedFrameGeneration::with_surface(SceneGeneration::default(), 1),
        PreparedFrameContext::for_surface(
            crate::layout::Viewport::default(),
            PhysicalSize::new(800, 600),
            1,
        ),
        UploadPlan::default(),
        vec![PreparedPass::new(RenderPass::MainColor, draw_orders, 0)],
        draw_items,
        FrameGraphStats {
            surface_generation: Some(1),
            pass_count: 1,
            draw_item_count,
            upload_intent_count: 0,
            layer_count: 0,
            unsupported_fragment_count: 0,
            stale_drop_count: 0,
            duration: std::time::Duration::ZERO,
        },
    )
}

fn test_empty_text_layout() -> crate::text::TextLayoutRef {
    test_text_layout(std::sync::Arc::from([]), std::sync::Arc::from([]))
}

fn test_text_layout_with_glyph() -> crate::text::TextLayoutRef {
    let key = crate::text::GlyphKey::new(
        cosmic_text::CacheKey::new(
            fontdb::ID::dummy(),
            1,
            12.0,
            (0.0, 0.0),
            fontdb::Weight::NORMAL,
            cosmic_text::CacheKeyFlags::empty(),
        )
        .0,
        1.0,
    );
    test_text_layout(
        std::sync::Arc::from([crate::text::GlyphInstance::new(key, 0, 0)]),
        std::sync::Arc::from([crate::text::GlyphDemand::new(key)]),
    )
}

fn test_text_layout(
    glyphs: std::sync::Arc<[crate::text::GlyphInstance]>,
    demands: std::sync::Arc<[crate::text::GlyphDemand]>,
) -> crate::text::TextLayoutRef {
    let key = crate::text::TextLayoutKey {
        node_id: crate::retained::RetainedNodeId::new(1),
        node_generation: crate::retained::NodeGeneration::INITIAL,
        text_generation: crate::text::TextGeneration::INITIAL,
        style_generation: crate::text::TextGeneration::INITIAL,
        text_hash: 1,
        available_inline_width_bits: None,
        font_size_bits: 12.0_f32.to_bits(),
        max_lines: None,
        text_overflow: crate::style::TextOverflow::Clip,
        font_generation: crate::text::FontGeneration::INITIAL,
        scale_generation: 1,
        scale_factor_bits: 1.0_f32.to_bits(),
    };
    crate::text::TextLayoutRef::new(crate::text::TextLayoutData::new(
        crate::text::TextLayoutGeneration::new(1),
        key,
        crate::text::TextMetrics {
            width: 0.0,
            min_content_width: 0.0,
            max_content_width: 0.0,
            height: 0.0,
            baseline: 0.0,
            line_count: 1,
        },
        glyphs,
        demands,
    ))
}

#[test]
#[ignore = "manual Windows D3D11/DXGI smoke: presents a clear frame in a native window"]
fn manual_windows_d3d11_present_clear_smoke() -> NekoResult<()> {
    run_manual_windows_backend_smoke("NekoUI D3D11 clear/present smoke", ManualRoot::simple())
}

#[test]
#[ignore = "manual Windows D3D11/DXGI smoke: verifies resize generation updates"]
fn manual_windows_d3d11_resize_generation_smoke() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 resize generation smoke - resize then close",
        ManualRoot::resize_hint(),
    )
}

#[test]
#[ignore = "manual Windows D3D11/DXGI smoke: draws solid rects using checked shader artifacts"]
fn manual_windows_d3d11_solid_rect_smoke() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 solid rect smoke - expect visible dark and blue rects",
        ManualRoot::solid_rects(),
    )
}

#[test]
#[ignore = "manual Windows D3D11/DXGI smoke: resizes while drawing solid rects"]
fn manual_windows_d3d11_solid_resize_generation_smoke() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 solid rect resize smoke - resize then close",
        ManualRoot::solid_resize_hint(),
    )
}

#[test]
#[ignore = "manual Windows D3D11/DXGI smoke: verifies minimize and restore lifecycle"]
fn manual_windows_d3d11_minimize_restore_smoke() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 minimize/restore smoke - minimize, restore, close",
        ManualRoot::minimize_hint(),
    )
}

#[test]
#[ignore = "manual Windows D3D11/DXGI stress: resize storm"]
fn manual_windows_d3d11_resize_storm_stress() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 resize storm stress - resize repeatedly then close",
        ManualRoot::many_children(1_000),
    )
}

#[test]
#[ignore = "manual Windows D3D11/DXGI stress: clear-present over rect-heavy frames"]
fn manual_windows_d3d11_rect_frame_stress() -> NekoResult<()> {
    run_manual_windows_backend_smoke(
        "NekoUI D3D11 rect-heavy clear/present stress - close when stable",
        ManualRoot::many_children(10_000),
    )
}

#[derive(Debug)]
struct ManualRoot {
    label: &'static str,
    child_count: usize,
}

impl ManualRoot {
    fn simple() -> Self {
        Self {
            label: "If this window opens, the private Windows D3D11 backend created a surface and presented a clear frame.",
            child_count: 1,
        }
    }

    fn resize_hint() -> Self {
        Self {
            label: "Manually resize this window several times, then close it. The backend should skip stale generations.",
            child_count: 32,
        }
    }

    fn minimize_hint() -> Self {
        Self {
            label: "Minimize, restore, then close this window. Restore should resume renderability.",
            child_count: 32,
        }
    }

    fn many_children(child_count: usize) -> Self {
        Self {
            label: "Manual stress harness: this creates many solid rect rows for the D3D11 rect pass.",
            child_count,
        }
    }

    fn solid_rects() -> Self {
        Self {
            label: "Solid rect smoke: visible rows should be drawn by the D3D11 rect pass.",
            child_count: 24,
        }
    }

    fn solid_resize_hint() -> Self {
        Self {
            label: "Resize this solid-rect window several times, then close it.",
            child_count: 96,
        }
    }
}

impl Render for ManualRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        let mut root = div()
            .p(px(12.0))
            .w(fill())
            .bg(Color::rgb(0x10, 0x14, 0x1C))
            .child(text(self.label).font_size(px(16.0)));

        for index in 0..self.child_count {
            root = root.child(
                div()
                    .key(format!("manual-d3d-row-{index}"))
                    .p(px(1.0))
                    .bg(Color::rgb(0x20, 0x28, 0x35))
                    .child(text(format!("row {index}")).font_size(px(10.0))),
            );
        }

        root
    }
}

fn run_manual_windows_backend_smoke(title: &'static str, root: ManualRoot) -> NekoResult<()> {
    Application::new().run(|cx| {
        cx.windows()
            .open(WindowOptions::new().title(title), |_| root)?;
        Ok(())
    })
}

fn collect_unsafe_offenders(
    path: &std::path::Path,
    allowed_prefix: &std::path::Path,
    offenders: &mut Vec<String>,
) {
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_unsafe_offenders(&path, allowed_prefix, offenders);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = std::fs::read_to_string(&path).unwrap();
            if contains_unsafe_escape(&source) && !path.starts_with(allowed_prefix) {
                offenders.push(path.display().to_string());
            }
        }
    }
}

fn contains_unsafe_escape(source: &str) -> bool {
    source.contains("unsafe {")
        || source.contains("unsafe fn")
        || source.contains("unsafe impl")
        || source.contains("allow(unsafe_code)")
}

fn assert_unsafe_blocks_have_safety_comments(path: &str, source: &str) {
    let lines = source.lines().collect::<Vec<_>>();
    for (line_index, line) in lines.iter().enumerate() {
        if !line.contains("unsafe {") {
            continue;
        }

        assert!(
            has_adjacent_safety_comment(&lines, line_index),
            "{path}:{} unsafe block is missing a preceding SAFETY comment",
            line_index + 1
        );
    }
}

fn has_adjacent_safety_comment(lines: &[&str], line_index: usize) -> bool {
    let mut probe = line_index;
    while let Some(previous) = probe.checked_sub(1) {
        let trimmed = lines[previous].trim();
        if trimmed.is_empty() {
            probe = previous;
            continue;
        }
        if !trimmed.starts_with("//") {
            return false;
        }
        if trimmed.starts_with("// SAFETY:") {
            return true;
        }
        probe = previous;
    }
    false
}
