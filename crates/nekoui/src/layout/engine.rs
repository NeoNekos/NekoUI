use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use crate::error::NekoError;
use crate::layout::{LayoutGeneration, LayoutTreeSnapshot, Viewport};
use crate::retained::RetainedLayoutInput;
use crate::text::{FontManager, TextMeasureStats};

use super::taffy_adapter;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LayoutPassStats {
    pub node_count: usize,
    pub changed_geometry_count: usize,
    pub duration: Duration,
    pub text_measure: TextMeasureStats,
    pub layout_deferred_count: u64,
    pub layout_blocked_on_text_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutPassOutput {
    pub snapshot: LayoutTreeSnapshot,
    pub stats: LayoutPassStats,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutPassFailure {
    error: NekoError,
    stats: Box<LayoutPassStats>,
}

impl LayoutPassFailure {
    pub(crate) fn new(error: NekoError, stats: LayoutPassStats) -> Self {
        Self {
            error,
            stats: Box::new(stats),
        }
    }

    pub(crate) fn error(&self) -> &NekoError {
        &self.error
    }

    pub(crate) fn stats(&self) -> &LayoutPassStats {
        self.stats.as_ref()
    }

    pub(crate) fn into_error(self) -> NekoError {
        self.error
    }
}

pub(crate) fn compute_layout(
    input: RetainedLayoutInput<'_>,
    viewport: Viewport,
    previous: Option<&LayoutTreeSnapshot>,
    font_manager: &FontManager,
) -> Result<LayoutPassOutput, LayoutPassFailure> {
    let started = std::time::Instant::now();
    let raw = match taffy_adapter::compute(input, viewport, font_manager) {
        Ok(raw) => raw,
        Err(taffy_adapter::RawLayoutError::Plain(error)) => {
            return Err(LayoutPassFailure::new(
                error,
                stats_from_text_measure(TextMeasureStats::default(), started.elapsed()),
            ));
        }
        Err(taffy_adapter::RawLayoutError::WithTextStats {
            error,
            text_measure,
        }) => {
            let stats = stats_from_text_measure(*text_measure, started.elapsed());
            return Err(LayoutPassFailure::new(error, stats));
        }
    };
    let changed_geometry_count = changed_geometry_count(previous, &raw.root);
    let generation = raw
        .root
        .as_ref()
        .map(|_| match previous.and_then(|old| old.generation()) {
            Some(old_generation)
                if changed_geometry_count == 0
                    && previous.is_some_and(|old| old.viewport() == viewport)
                    && previous
                        .is_some_and(|old| old.retained_generation() == input.generation()) =>
            {
                old_generation
            }
            Some(old_generation) => old_generation.next(),
            None => LayoutGeneration::INITIAL,
        });
    let node_count = raw.node_count;
    let snapshot = LayoutTreeSnapshot::new(
        generation,
        input.generation(),
        viewport,
        node_count,
        raw.root,
    );

    Ok(LayoutPassOutput {
        snapshot,
        stats: LayoutPassStats {
            node_count,
            changed_geometry_count,
            duration: started.elapsed(),
            layout_deferred_count: raw.text_measure.deferred_count,
            layout_blocked_on_text_count: raw.text_measure.blocked_count,
            text_measure: raw.text_measure,
        },
    })
}

fn stats_from_text_measure(
    text_measure: TextMeasureStats,
    duration: std::time::Duration,
) -> LayoutPassStats {
    LayoutPassStats {
        node_count: 0,
        changed_geometry_count: 0,
        duration,
        layout_deferred_count: text_measure.deferred_count,
        layout_blocked_on_text_count: text_measure.blocked_count,
        text_measure,
    }
}

fn changed_geometry_count(
    previous: Option<&LayoutTreeSnapshot>,
    root: &Option<crate::layout::LayoutNodeSnapshot>,
) -> usize {
    let Some(previous) = previous else {
        return root.as_ref().map_or(0, count_subtree);
    };
    let Some(root) = root else {
        return previous.node_count();
    };

    let previous_geometry = previous.geometry().into_iter().collect::<BTreeMap<_, _>>();
    let mut current_geometry = Vec::new();
    root.collect_geometry(&mut current_geometry);

    let current_ids = current_geometry
        .iter()
        .map(|(node_id, _)| *node_id)
        .collect::<BTreeSet<_>>();
    let changed_or_added = current_geometry
        .into_iter()
        .filter(|(node_id, geometry)| previous_geometry.get(node_id) != Some(geometry))
        .count();
    let removed = previous_geometry
        .keys()
        .filter(|node_id| !current_ids.contains(node_id))
        .count();

    changed_or_added + removed
}

fn count_subtree(node: &crate::layout::LayoutNodeSnapshot) -> usize {
    1 + node.children().iter().map(count_subtree).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use crate::element::ElementKind;
    use crate::layout::snapshot::LayoutBoxes;
    use crate::layout::{LayoutNodeSnapshot, LayoutRect, LayoutSize, LayoutTreeSnapshot, Viewport};
    use crate::retained::{RetainedNodeId, RetainedTreeGeneration};

    use super::changed_geometry_count;

    #[test]
    fn changed_geometry_counts_removed_layout_nodes() {
        let previous = LayoutTreeSnapshot::new(
            None,
            Some(RetainedTreeGeneration::INITIAL),
            Viewport::default(),
            2,
            Some(node(
                1,
                LayoutRect::new(0.0, 0.0, 100.0, 100.0),
                vec![node(2, LayoutRect::new(0.0, 0.0, 10.0, 10.0), Vec::new())],
            )),
        );
        let current = Some(node(1, LayoutRect::new(0.0, 0.0, 100.0, 100.0), Vec::new()));

        assert_eq!(changed_geometry_count(Some(&previous), &current), 1);
    }

    fn node(id: u64, rect: LayoutRect, children: Vec<LayoutNodeSnapshot>) -> LayoutNodeSnapshot {
        LayoutNodeSnapshot::new(
            RetainedNodeId::new(id),
            ElementKind::Div,
            None,
            LayoutBoxes {
                margin_rect: rect,
                border_rect: rect,
                padding_rect: rect,
                content_rect: rect,
                content_size: LayoutSize::new(rect.width(), rect.height()),
                text_layout: None,
            },
            children,
        )
    }
}
