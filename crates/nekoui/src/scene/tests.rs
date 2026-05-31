use crate::app::{Context, Render};
use crate::diagnostic::DirtyLane;
use crate::element::{Element, ElementKind, IntoElement, div, input, text};
use crate::interaction::{PointerInput, ScrollDelta, ScrollPhase, WheelInput};
use crate::layout::{LayoutPoint, LayoutSize, text_viewport_placement};
use crate::platform::PlatformFact;
use crate::runtime::Runtime;
use crate::scene::{
    DamageReason, PaintFragmentKind, ResourceDemandKind, SceneCompileInput, compile_scene,
    scene_generation_for_inputs, scene_publish_is_current,
};
use crate::style::{Color, Display, Overflow, StyleExt, opacity, px};
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

fn compile_root(root: impl crate::element::IntoElement) -> crate::scene::PaintScene {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(root))
        .unwrap();
    runtime.scene_snapshot(window).unwrap()
}

#[test]
fn deterministic_paint_order_is_background_text_children() {
    let scene = compile_root(
        div()
            .key("root")
            .bg(Color::rgb(1, 2, 3))
            .child(text("A").key("a").text_color(Color::rgb(4, 5, 6)))
            .child(text("B").key("b").text_color(Color::rgb(7, 8, 9))),
    );
    let kinds = scene
        .fragments()
        .iter()
        .map(|fragment| fragment.kind())
        .collect::<Vec<_>>();

    assert!(matches!(kinds[0], PaintFragmentKind::Rect { .. }));
    assert!(matches!(kinds[1], PaintFragmentKind::Text { .. }));
    assert!(matches!(kinds[2], PaintFragmentKind::Text { .. }));
    assert!(scene.fragments()[0].order().raw() < scene.fragments()[1].order().raw());
    assert!(scene.fragments()[1].order().raw() < scene.fragments()[2].order().raw());
}

#[test]
fn display_none_and_opacity_zero_obey_visibility_baseline() {
    let scene = compile_root(
        div()
            .key("root")
            .child(text("visible").key("visible"))
            .child(text("none").key("none").display(Display::None))
            .child(text("transparent").key("transparent").opacity(opacity(0.0))),
    );

    assert_eq!(scene.fragments().len(), 1);
    assert_eq!(scene.hit_test().entries().len(), 3);
}

#[test]
fn hit_test_order_tracks_scene_order_for_later_topmost_targeting() {
    let scene = compile_root(
        div()
            .key("root")
            .child(text("first").key("first"))
            .child(text("second").key("second")),
    );
    let entries = scene.hit_test().entries();

    assert_eq!(entries.len(), 3);
    assert!(entries[0].order().raw() < entries[1].order().raw());
    assert!(entries[1].order().raw() < entries[2].order().raw());
}

#[test]
fn hit_test_metadata_matches_retained_and_layout_nodes() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(text("target").key("target")))
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let scene = runtime.scene_snapshot(window).unwrap();
    let retained_target = retained.find_by_key("target").unwrap();
    let layout_target = layout.find_by_key("target").unwrap();
    let hit_target = scene
        .hit_test()
        .entries()
        .iter()
        .find(|entry| entry.node_id() == retained_target.id())
        .unwrap();

    assert_eq!(hit_target.node_generation(), retained_target.generation());
    assert_eq!(hit_target.rect(), layout_target.border_rect());
    let path = hit_target.path();
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].node_id(), retained.root().unwrap().id());
    assert_eq!(path[1].node_id(), retained_target.id());
}

#[test]
fn overflow_scroll_style_resolves_and_layout_exposes_scroll_metadata() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("scroll")
                    .w(px(100.0))
                    .h(px(80.0))
                    .overflow(Overflow::Scroll)
                    .child(text("content").key("content").h(px(200.0))),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let retained_scroll = retained.find_by_key("scroll").unwrap();
    let layout_scroll = layout.find_by_key("scroll").unwrap();

    assert_eq!(
        retained_scroll.resolved_style().layout().overflow(),
        Overflow::Scroll
    );
    assert_eq!(layout_scroll.scroll().overflow(), Overflow::Scroll);
    assert_eq!(layout_scroll.scroll().viewport().height(), 80.0);
    assert_eq!(layout_scroll.scroll().content_extent().height(), 200.0);
    assert_eq!(layout_scroll.scroll().max_offset().y(), 120.0);
}

#[test]
fn clipped_out_scrolled_child_hit_test_changes_after_scroll() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("scroll")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(text("top").key("top").h(px(100.0)))
                    .child(text("bottom").key("bottom").h(px(100.0))),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let bottom_id = retained.find_by_key("bottom").unwrap().id();
    let scene = runtime.scene_snapshot(window).unwrap();
    let bottom_entry = scene
        .hit_test()
        .entries()
        .iter()
        .find(|entry| entry.node_id() == bottom_id)
        .unwrap();
    assert!(bottom_entry.clip().is_some());
    assert_ne!(
        scene
            .hit_test()
            .hit_test(LayoutPoint::new(1.0, 101.0))
            .map(|entry| entry.node_id()),
        Some(bottom_id)
    );

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 100.0), ScrollPhase::Moved),
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let scrolled_scene = runtime.scene_snapshot(window).unwrap();
    let target = scrolled_scene
        .hit_test()
        .hit_test(LayoutPoint::new(1.0, 1.0))
        .unwrap();
    assert_eq!(target.node_id(), bottom_id);
}

#[test]
fn hit_test_returns_topmost_entry_containing_point() {
    let scene = compile_root(
        div()
            .key("root")
            .child(text("first").key("first"))
            .child(text("second").key("second")),
    );
    let expected = scene.hit_test().entries()[2].clone();
    let point =
        crate::layout::LayoutPoint::new(expected.rect().x() + 1.0, expected.rect().y() + 1.0);
    let target = scene.hit_test().hit_test(point).unwrap();

    assert_eq!(target.order(), expected.order());
}

#[test]
fn hit_test_skips_display_none_but_keeps_opacity_zero() {
    let scene = compile_root(
        div()
            .key("root")
            .child(text("none").key("none").display(Display::None))
            .child(text("transparent").key("transparent").opacity(opacity(0.0))),
    );
    let target = scene
        .hit_test()
        .hit_test(crate::layout::LayoutPoint::new(1.0, 1.0))
        .unwrap();

    assert_eq!(target.order(), scene.hit_test().entries()[1].order());
}

#[test]
fn opacity_zero_parent_suppresses_subtree_paint_but_keeps_hit_test_entries() {
    let scene = compile_root(
        div().key("root").child(
            div()
                .key("transparent-parent")
                .bg(Color::rgb(1, 2, 3))
                .opacity(opacity(0.0))
                .child(text("transparent child").key("transparent-child")),
        ),
    );

    let entries = scene.hit_test().entries();

    assert_eq!(scene.fragments().len(), 0);
    assert_eq!(entries.len(), 3);
    assert!(entries[1].order().raw() < entries[2].order().raw());
    let point =
        crate::layout::LayoutPoint::new(entries[2].rect().x() + 1.0, entries[2].rect().y() + 1.0);
    let target = scene.hit_test().hit_test(point).unwrap();

    assert_eq!(target.order(), entries[2].order());
}

#[test]
fn generation_key_matches_input_generations() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(text("A")))
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let style = runtime.style_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let scene = runtime.scene_snapshot(window).unwrap();

    assert_eq!(
        scene.generation(),
        scene_generation_for_inputs(&retained, &style, &layout)
    );
    assert_eq!(
        scene.generation().retained_generation(),
        retained.generation()
    );
    assert_eq!(scene.generation().layout_generation(), layout.generation());
    assert_eq!(
        scene.generation().viewport_generation(),
        layout.viewport().generation().raw()
    );
    assert!(!scene.generation().style_generation().facts().is_empty());
    assert!(!scene.generation().text_generation().facts().is_empty());
}

#[test]
fn text_glyph_demands_match_scene_text_generation() {
    let scene = compile_root(div().key("root").child(text("Glyphs").key("label")));
    let glyph_demand = scene
        .resource_demands()
        .iter()
        .find(|demand| demand.kind() == ResourceDemandKind::Glyph)
        .unwrap();

    assert_eq!(
        glyph_demand.expected_generation(),
        scene.generation().text_generation()
    );
    assert!(glyph_demand.glyphs().is_some());
    let text_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| matches!(fragment.kind(), PaintFragmentKind::Text { .. }))
        .unwrap();
    assert!(text_fragment.text_layout().is_some());
}

#[test]
fn focused_tall_input_text_and_caret_share_centered_line_box_origin() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .child(input("hi").key("field").h(px(48.0)).font_size(px(12.0)))
                    .child(text("plain").key("label").h(px(48.0)).font_size(px(12.0))),
            )
        })
        .unwrap();
    let unfocused_generation = runtime.scene_snapshot(window).unwrap().generation();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let scene = runtime.scene_snapshot(window).unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let field = retained.find_by_key("field").unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let field_text_layout = field_layout.text_layout().unwrap();
    let field_content = field_layout.content_rect();
    let field_line_box_y = field_content.y()
        + ((field_content.height() - field_text_layout.metrics().height) / 2.0).max(0.0);
    let field_text_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == field.id()
                && matches!(fragment.kind(), PaintFragmentKind::Text { .. })
        })
        .expect("focused input should retain its text fragment");
    let expected_caret = field_text_layout
        .trailing_caret_rect()
        .translate(field_content.x(), field_line_box_y);
    let caret_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == field.id()
                && matches!(fragment.kind(), PaintFragmentKind::Rect { color } if *color == Color::BLACK)
                && fragment.rect() == expected_caret
        })
        .expect("focused input should emit a static caret rect");
    let label = retained.find_by_key("label").unwrap();
    let label_layout = layout.find_by_key("label").unwrap();
    let label_text_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == label.id()
                && matches!(fragment.kind(), PaintFragmentKind::Text { .. })
        })
        .expect("ordinary text should emit a text fragment");

    assert_eq!(field_text_fragment.rect().x(), field_content.x());
    assert_eq!(field_text_fragment.rect().y(), field_line_box_y);
    assert!(field_text_fragment.rect().y() > field_content.y());
    assert_eq!(caret_fragment.rect(), expected_caret);
    assert_eq!(
        caret_fragment.rect().y() - field_text_fragment.rect().y(),
        field_text_layout.trailing_caret_rect().y()
    );
    assert!(
        scene
            .fragments()
            .iter()
            .position(|fragment| fragment == field_text_fragment)
            .unwrap()
            < scene
                .fragments()
                .iter()
                .position(|fragment| {
                    fragment.node_id() == field.id()
                        && matches!(fragment.kind(), PaintFragmentKind::Rect { color } if *color == Color::BLACK)
                        && fragment.rect() == expected_caret
                })
                .unwrap()
    );
    assert_eq!(label_text_fragment.rect(), label_layout.content_rect());
    assert_ne!(scene.generation(), unfocused_generation);
}

#[test]
fn ordinary_tall_text_uses_content_top_origin() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .child(text("plain").key("label").h(px(48.0)).font_size(px(12.0))),
            )
        })
        .unwrap();

    let scene = runtime.scene_snapshot(window).unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let label = retained.find_by_key("label").unwrap();
    let label_layout = layout.find_by_key("label").unwrap();
    let label_text = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == label.id()
                && matches!(fragment.kind(), PaintFragmentKind::Text { .. })
        })
        .unwrap();

    assert!(
        ((label_layout.content_rect().height()
            - label_layout.text_layout().unwrap().metrics().height)
            / 2.0)
            .max(0.0)
            > 0.0
    );
    assert_eq!(label_text.rect(), label_layout.content_rect());
}

#[test]
fn focused_input_emits_static_caret_rect_after_text_fragment() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();
    let unfocused_generation = runtime.scene_snapshot(window).unwrap().generation();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let scene = runtime.scene_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let text_layout = field_layout.text_layout().unwrap();
    let expected_caret =
        text_viewport_placement(ElementKind::Input, field_layout.content_rect(), text_layout)
            .visible_caret_rect();
    let field_fragments = scene
        .fragments()
        .iter()
        .filter(|fragment| {
            fragment
                .rect()
                .intersect(field_layout.content_rect())
                .is_some()
        })
        .collect::<Vec<_>>();
    let text_index = field_fragments
        .iter()
        .position(|fragment| matches!(fragment.kind(), PaintFragmentKind::Text { .. }))
        .expect("focused input should retain its text fragment");
    let caret_index = field_fragments
        .iter()
        .position(|fragment| {
            matches!(fragment.kind(), PaintFragmentKind::Rect { color } if *color == Color::BLACK)
                && fragment.rect() == expected_caret
        })
        .expect("focused input should emit a static caret rect");

    assert!(text_index < caret_index);
    assert_ne!(scene.generation(), unfocused_generation);
}

#[test]
fn focused_long_input_reveals_text_and_caret_with_shared_inline_scroll() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    input("AAAA AAAA AAAA AAAA")
                        .key("field")
                        .w(px(36.0))
                        .font_size(px(12.0)),
                ),
            )
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let scene = runtime.scene_snapshot(window).unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let field = retained.find_by_key("field").unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let content = field_layout.content_rect();
    let text_layout = field_layout.text_layout().unwrap();
    let placement = text_viewport_placement(ElementKind::Input, content, text_layout);
    let text_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == field.id()
                && matches!(fragment.kind(), PaintFragmentKind::Text { .. })
        })
        .expect("focused input should retain its text fragment");
    let caret_fragment = scene
        .fragments()
        .iter()
        .find(|fragment| {
            fragment.node_id() == field.id()
                && matches!(fragment.kind(), PaintFragmentKind::Rect { color } if *color == Color::BLACK)
                && fragment.rect() == placement.visible_caret_rect()
        })
        .expect("focused input should emit a visible caret rect");

    assert_eq!(text_layout.metrics().line_count, 1);
    assert!(text_layout.trailing_caret_rect().x() > content.width());
    assert!(placement.input_inline_scroll() > 0.0);
    assert_eq!(placement.viewport_rect(), content);
    assert_eq!(text_fragment.rect(), placement.text_draw_rect());
    assert_eq!(text_fragment.clip(), Some(content));
    assert_eq!(caret_fragment.rect(), placement.visible_caret_rect());
    assert_eq!(
        content.x() - text_fragment.rect().x(),
        placement.input_inline_scroll()
    );
    assert!(caret_fragment.rect().x() >= content.x());
    assert!(
        caret_fragment.rect().x() + caret_fragment.rect().width() <= content.x() + content.width()
    );
}

#[test]
fn partial_opacity_emits_unsupported_fragment_demand_and_diagnostic() {
    let scene = compile_root(div().key("root").child(text("layer").opacity(opacity(0.5))));

    assert!(scene.fragments().iter().any(|fragment| matches!(
        fragment.kind(),
        PaintFragmentKind::Unsupported {
            capability: "partial_opacity_layer"
        }
    )));
    assert!(scene.resource_demands().iter().any(|demand| {
        demand.kind() == ResourceDemandKind::Unsupported
            && demand.expected_generation() == scene.generation().style_generation()
    }));
    assert_eq!(scene.stats().unsupported_fragment_count, 1);
    assert!(scene.diagnostics().iter().any(|diagnostic| {
        diagnostic.message() == "unsupported scene fragment capability" && diagnostic.count() == 1
    }));
}

#[test]
fn damage_reasons_are_value_level_observable() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(
            WindowOptions::new().logical_size(LayoutSize::new(200.0, 100.0)),
            |_| TestRoot::new(div().key("root")),
        )
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let style = runtime.style_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let initial_scene = runtime.scene_snapshot(window).unwrap();

    assert_eq!(initial_scene.damage().reason(), DamageReason::Initial);

    let unchanged = compile_scene(SceneCompileInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction: None,
        previous: Some(&initial_scene),
    });
    assert_eq!(unchanged.scene.damage().reason(), DamageReason::Unchanged);
    assert_eq!(unchanged.scene.damage().region_count(), 0);

    let output = compile_scene(SceneCompileInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction: None,
        previous: None,
    });
    assert_eq!(output.scene.damage().reason(), DamageReason::Initial);
    assert!(output.scene.damage().region_count() > 0);
}

#[test]
fn publish_gate_rejects_output_or_current_snapshot_generation_mismatch() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(text("A")))
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let style = runtime.style_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let expected = scene_generation_for_inputs(&retained, &style, &layout);
    let first_scene = runtime.scene_snapshot(window).unwrap();

    let mut other_runtime = Runtime::new();
    let other_window = other_runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(text("B")))
        })
        .unwrap();
    let current_retained = other_runtime.retained_snapshot(other_window).unwrap();
    let current_style = other_runtime.style_snapshot(other_window).unwrap();
    let current_layout = other_runtime.layout_snapshot(other_window).unwrap();
    let current_scene = other_runtime.scene_snapshot(other_window).unwrap();

    assert!(!scene_publish_is_current(
        expected.clone(),
        first_scene.generation(),
        &current_retained,
        &current_style,
        &current_layout,
    ));
    assert!(!scene_publish_is_current(
        expected,
        current_scene.generation(),
        &retained,
        &style,
        &layout,
    ));
}

#[test]
fn stale_like_input_change_produces_conservative_damage() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(
            WindowOptions::new().logical_size(LayoutSize::new(200.0, 100.0)),
            |_| TestRoot::new(div().key("root")),
        )
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let style = runtime.style_snapshot(window).unwrap();
    let layout = runtime.layout_snapshot(window).unwrap();
    let previous = runtime.scene_snapshot(window).unwrap();

    runtime
        .resize_window(window, LayoutSize::new(300.0, 100.0))
        .unwrap();
    let resized_layout = runtime.layout_snapshot(window).unwrap();
    let output = compile_scene(SceneCompileInput {
        retained: &retained,
        style: &style,
        layout: &resized_layout,
        interaction: None,
        previous: Some(&previous),
    });

    assert_eq!(layout.generation().unwrap().raw(), 1);
    assert_eq!(
        output.scene.damage().reason(),
        DamageReason::ConservativeInputChange
    );
    assert_eq!(output.scene.damage().region_count(), 1);
    assert!(output.scene.damage().total_area() >= 30_000.0);
}

#[test]
fn diagnostics_counters_are_aggregate_not_per_fragment() {
    let mut runtime = Runtime::new();
    runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().child(text("A")).child(text("B")))
        })
        .unwrap();
    let report = runtime.performance_report();
    let diagnostics = runtime.diagnostics().snapshot();

    assert_eq!(report.scene.compile_count, 1);
    assert_eq!(
        report.scene.last_compile.fragment_count,
        report.scene.fragment_count
    );
    assert_eq!(report.scene.published_node_count, report.scene.node_count);
    assert!(report.scene.last_compile.fragment_count >= 2);
    assert!(diagnostics.counter("scene.compile") >= 1);
    assert!(diagnostics.counter("scene.fragment_count") >= 2);
    assert!(
        runtime
            .performance_report()
            .dirty_lanes
            .iter()
            .any(|window| window.lanes.contains(DirtyLane::Paint.flag()))
    );
}
