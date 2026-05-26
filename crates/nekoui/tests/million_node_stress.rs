use nekoui::layout::LayoutSize;
use nekoui::prelude::*;

#[derive(Clone, Copy, Debug)]
enum StressShape {
    Simple,
    Complex,
}

#[derive(Debug)]
struct StressRoot {
    root: Entity<nekoui::Element>,
}

impl Render for StressRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.root.read(cx, Clone::clone).unwrap()
    }
}

// Manual stress command:
// cargo test -p nekoui --test million_node_stress -- --ignored --nocapture
#[test]
#[ignore = "manual million-node retained/style stress harness"]
fn million_node_retained_style_stress() {
    fn branch(depth: usize, width: usize, index: usize, style_version: u8) -> nekoui::Element {
        if depth == 0 {
            return text(format!("leaf-{index}"))
                .key(format!("leaf-{index}"))
                .font_size(px(12.0 + f32::from(style_version)))
                .into_element();
        }

        let mut node = div()
            .key(format!("node-{depth}-{index}"))
            .p(px((depth % 5) as f32 + f32::from(style_version)))
            .bg(Color::rgb(
                (depth % 255) as u8,
                (index % 255) as u8,
                64 + style_version,
            ));
        for child in 0..width {
            node = node.child(branch(
                depth - 1,
                width,
                index * width + child,
                style_version,
            ));
        }
        node.into_element()
    }

    fn pruned_branch(depth: usize, width: usize, index: usize) -> nekoui::Element {
        if depth == 0 {
            return text(format!("leaf-{index}"))
                .key(format!("leaf-{index}"))
                .into_element();
        }

        let mut node = div()
            .key(format!("node-{depth}-{index}"))
            .p(px((depth % 5) as f32));
        for child in 0..width - 1 {
            node = node.child(pruned_branch(depth - 1, width, index * width + child));
        }
        node.into_element()
    }

    let width = 10;
    let depth = 6;
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| branch(depth, width, 1, 0));
            cx.windows()
                .open(WindowOptions::new().title("Million Node Stress"), |_| {
                    StressRoot { root: root.clone() }
                })?;
            let first = cx.performance_report().retained.last_diff.clone();
            assert!(first.created >= 1_000_000);
            assert_eq!(first.preserved, 0);
            assert_eq!(first.destroyed, 0);

            root.update(cx, |root, cx| {
                *root = branch(depth, width, 1, 1);
                cx.notify();
                Ok(())
            })?;
            let update = cx.performance_report().retained.last_diff.clone();
            assert!(update.preserved >= 1_000_000);
            assert_eq!(update.created, 0);
            assert_eq!(update.replaced, 0);
            assert_eq!(update.destroyed, 0);

            root.update(cx, |root, cx| {
                *root = pruned_branch(depth, width, 1);
                cx.notify();
                Ok(())
            })?;
            let delete = cx.performance_report().retained.last_diff.clone();
            assert!(delete.preserved >= 100_000);
            assert!(delete.destroyed > 0);
            assert_eq!(delete.created, 0);
            Ok(())
        })
        .unwrap();

    let report = run.performance_report();
    assert!(report.retained.last_diff.destroyed > 0);
    assert!(report.retained.node_count > 100_000);
    assert!(report.retained.node_count < 1_000_000);
    println!("{report:#?}");
}

fn stress_branch(depth: usize, width: usize, index: usize, shape: StressShape) -> nekoui::Element {
    if depth == 0 {
        let text = text(format!("leaf-{index}")).key(format!("leaf-{index}"));
        return match shape {
            StressShape::Simple => text.into_element(),
            StressShape::Complex => text.font_size(px(12.0 + (index % 3) as f32)).into_element(),
        };
    }

    let mut node = div().key(format!("node-{depth}-{index}"));
    if matches!(shape, StressShape::Complex) {
        node = node
            .p(px((depth % 5) as f32))
            .gap(px((index % 3) as f32))
            .bg(Color::rgb((depth % 255) as u8, (index % 255) as u8, 96));
    }
    for child in 0..width {
        node = node.child(stress_branch(
            depth - 1,
            width,
            index * width + child,
            shape,
        ));
    }
    node.into_element()
}

fn run_layout_stress(depth: usize, width: usize, shape: StressShape, minimum_nodes: usize) {
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| stress_branch(depth, width, 1, shape));
            cx.windows()
                .open(WindowOptions::new(), |_| StressRoot { root: root.clone() })?;
            Ok(())
        })
        .unwrap();
    let report = run.performance_report();

    assert!(report.layout.node_count >= minimum_nodes);
    assert_eq!(report.layout.pass_count, 1);
    println!(
        "{shape:?} layout stress nodes: {}",
        report.layout.node_count
    );
}

fn run_scene_stress(depth: usize, width: usize, shape: StressShape, minimum_nodes: usize) {
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| stress_branch(depth, width, 1, shape));
            cx.windows()
                .open(WindowOptions::new(), |_| StressRoot { root: root.clone() })?;
            Ok(())
        })
        .unwrap();
    let report = run.performance_report();

    assert!(report.scene.node_count >= minimum_nodes);
    assert!(report.scene.compile_count >= 1);
    assert!(report.scene.last_compile.fragment_count > 0);
    assert_eq!(
        report.scene.fragment_count,
        report.scene.last_compile.fragment_count
    );
    assert!(report.scene.hit_test_entry_count >= minimum_nodes);
    assert!(report.phase_durations.contains_key("scene.compile"));
    println!(
        "{shape:?} scene stress nodes={} fragments={} hit_entries={} damage_regions={} resource_demands={}",
        report.scene.node_count,
        report.scene.fragment_count,
        report.scene.hit_test_entry_count,
        report.scene.damage_region_count,
        report.scene.resource_demand_count,
    );
}

#[test]
#[ignore = "manual 10k simple layout stress harness"]
fn ten_thousand_simple_layout_stress() {
    run_layout_stress(4, 10, StressShape::Simple, 10_000);
}

#[test]
#[ignore = "manual 10k complex layout stress harness"]
fn ten_thousand_complex_layout_stress() {
    run_layout_stress(4, 10, StressShape::Complex, 10_000);
}

#[test]
#[ignore = "manual 100k simple layout stress harness"]
fn hundred_thousand_simple_layout_stress() {
    run_layout_stress(5, 10, StressShape::Simple, 100_000);
}

#[test]
#[ignore = "manual 100k complex layout stress harness"]
fn hundred_thousand_complex_layout_stress() {
    run_layout_stress(5, 10, StressShape::Complex, 100_000);
}

#[test]
#[ignore = "manual 1m simple layout stress harness"]
fn million_simple_layout_stress() {
    run_layout_stress(6, 10, StressShape::Simple, 1_000_000);
}

#[test]
#[ignore = "manual 1m complex layout stress harness"]
fn million_complex_layout_stress() {
    run_layout_stress(6, 10, StressShape::Complex, 1_000_000);
}

#[test]
#[ignore = "manual 10k simple scene stress harness"]
fn ten_thousand_simple_scene_stress() {
    run_scene_stress(4, 10, StressShape::Simple, 10_000);
}

#[test]
#[ignore = "manual 10k complex scene stress harness"]
fn ten_thousand_complex_scene_stress() {
    run_scene_stress(4, 10, StressShape::Complex, 10_000);
}

#[test]
#[ignore = "manual 100k simple scene stress harness"]
fn hundred_thousand_simple_scene_stress() {
    run_scene_stress(5, 10, StressShape::Simple, 100_000);
}

#[test]
#[ignore = "manual 100k complex scene stress harness"]
fn hundred_thousand_complex_scene_stress() {
    run_scene_stress(5, 10, StressShape::Complex, 100_000);
}

#[test]
#[ignore = "manual 1m simple scene stress harness"]
fn million_simple_scene_stress() {
    run_scene_stress(6, 10, StressShape::Simple, 1_000_000);
}

#[test]
#[ignore = "manual 1m complex scene stress harness"]
fn million_complex_scene_stress() {
    run_scene_stress(6, 10, StressShape::Complex, 1_000_000);
}

fn text_heavy_branch(depth: usize, width: usize, index: usize, text_len: usize) -> nekoui::Element {
    if depth == 0 {
        let payload = format!("leaf-{index}-{}", "text".repeat(text_len));
        return text(payload)
            .key(format!("text-leaf-{index}"))
            .font_size(px(12.0 + (index % 5) as f32))
            .into_element();
    }

    let mut node = div()
        .key(format!("text-node-{depth}-{index}"))
        .w(fill())
        .p(px((depth % 3) as f32));
    for child in 0..width {
        node = node.child(text_heavy_branch(
            depth - 1,
            width,
            index * width + child,
            text_len,
        ));
    }
    node.into_element()
}

fn run_text_heavy_stress(depth: usize, width: usize, minimum_nodes: usize, label: &str) {
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| text_heavy_branch(depth, width, 1, 6));
            cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(640.0, 480.0)),
                |_| StressRoot { root: root.clone() },
            )?;
            Ok(())
        })
        .unwrap();
    let report = run.performance_report();

    assert!(report.layout.node_count >= minimum_nodes);
    assert!(report.scene.node_count >= minimum_nodes);
    assert!(report.scene.compile_count >= 1);
    assert!(report.scene.fragment_count > 0);
    assert!(report.scene.hit_test_entry_count >= minimum_nodes);
    assert!(report.phase_durations.contains_key("scene.compile"));
    assert!(report.scene.resource_demand_count > 0);
    assert!(report.layout.text_query_count > 0);
    assert!(report.text.measure_count > 0);
    println!(
        "{label} text-heavy stress nodes={} scene_fragments={} scene_resource_demands={} text_queries={} text_measures={} text_cache_hits={} text_cache_misses={}",
        report.layout.node_count,
        report.scene.fragment_count,
        report.scene.resource_demand_count,
        report.layout.text_query_count,
        report.text.measure_count,
        report.text.cache_hits,
        report.text.cache_misses,
    );
}

#[test]
#[ignore = "manual 10k text-heavy layout stress harness"]
fn ten_thousand_text_heavy_layout_stress() {
    run_text_heavy_stress(4, 10, 10_000, "10k");
}

#[test]
#[ignore = "manual 100k text-heavy layout stress harness"]
fn hundred_thousand_text_heavy_layout_stress() {
    run_text_heavy_stress(5, 10, 100_000, "100k");
}

#[test]
#[ignore = "manual 1m text-heavy layout stress harness"]
fn million_text_heavy_layout_stress() {
    run_text_heavy_stress(6, 10, 1_000_000, "1m");
}
