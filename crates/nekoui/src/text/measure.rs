use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use crate::layout::LayoutRect;
use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::style::{ResolvedTextStyle, TextOverflow};
use crate::text::{
    FontGeneration, FontManager, GlyphDemand, GlyphInstance, GlyphKey, TextLayoutData,
    TextLayoutGeneration, TextLayoutKey, TextLayoutRef,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct TextGeneration(u64);

impl TextGeneration {
    pub(crate) const INITIAL: Self = Self(1);

    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextMeasureDependencyKind {
    FontGenerationStale,
    FontFallbackUnavailable,
    ShapingUnavailable,
}

impl TextMeasureDependencyKind {
    const ALL: [Self; 3] = [
        Self::FontGenerationStale,
        Self::FontFallbackUnavailable,
        Self::ShapingUnavailable,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::FontGenerationStale => "font_generation_stale",
            Self::FontFallbackUnavailable => "font_fallback_unavailable",
            Self::ShapingUnavailable => "shaping_unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextMeasureErrorKind {
    InvalidStyle,
    ShapingFailed,
}

impl TextMeasureErrorKind {
    const ALL: [Self; 2] = [Self::InvalidStyle, Self::ShapingFailed];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidStyle => "invalid_style",
            Self::ShapingFailed => "shaping_failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextLayoutMode {
    SoftWrap,
    SingleLineInput,
}

impl TextLayoutMode {
    const fn rank(self) -> u8 {
        match self {
            Self::SoftWrap => 0,
            Self::SingleLineInput => 1,
        }
    }

    fn wrap(self) -> cosmic_text::Wrap {
        match self {
            Self::SoftWrap => cosmic_text::Wrap::WordOrGlyph,
            Self::SingleLineInput => cosmic_text::Wrap::None,
        }
    }

    fn height_limit(self, line_height: f32) -> Option<f32> {
        match self {
            Self::SoftWrap => None,
            Self::SingleLineInput => Some(line_height),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextWrapPolicy {
    SoftWrap,
}

impl TextWrapPolicy {
    const DEFAULT: Self = Self::SoftWrap;

    const fn rank(self) -> u8 {
        match self {
            Self::SoftWrap => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextWhiteSpacePolicy {
    Normal,
}

impl TextWhiteSpacePolicy {
    const DEFAULT: Self = Self::Normal;

    const fn rank(self) -> u8 {
        match self {
            Self::Normal => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextLineHeightPolicy {
    Default,
}

impl TextLineHeightPolicy {
    const DEFAULT: Self = Self::Default;

    const fn rank(self) -> u8 {
        match self {
            Self::Default => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextMetrics {
    pub(crate) width: f32,
    pub(crate) min_content_width: f32,
    pub(crate) max_content_width: f32,
    pub(crate) height: f32,
    pub(crate) baseline: f32,
    pub(crate) line_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureDependency {
    kind: TextMeasureDependencyKind,
    reason: Cow<'static, str>,
}

impl TextMeasureDependency {
    pub(crate) fn new(
        kind: TextMeasureDependencyKind,
        reason: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }

    pub(crate) fn kind(&self) -> TextMeasureDependencyKind {
        self.kind
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureError {
    kind: TextMeasureErrorKind,
    message: Cow<'static, str>,
}

impl TextMeasureError {
    pub(crate) fn new(kind: TextMeasureErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn kind(&self) -> TextMeasureErrorKind {
        self.kind
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TextMeasureResult {
    Ready(TextMetrics),
    Deferred(TextMeasureDependency),
    Failed(TextMeasureError),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TextLayoutResult {
    Ready(TextLayoutRef),
    Deferred(TextMeasureDependency),
    Failed(TextMeasureError),
}

#[derive(Clone, Debug, PartialEq)]
struct TextMeasureReady {
    layout: TextLayoutRef,
}

impl TextMeasureReady {
    fn layout(self) -> TextLayoutRef {
        self.layout
    }
}

#[derive(Clone, Debug, PartialEq)]
enum TextMeasureOutcome {
    Ready(TextMeasureReady),
    Deferred(TextMeasureDependency),
    Failed(TextMeasureError),
}

impl TextMeasureOutcome {
    fn map_ready<T>(self, map: impl FnOnce(TextMeasureReady) -> T) -> TextReadyResult<T> {
        match self {
            Self::Ready(ready) => TextReadyResult::Ready(map(ready)),
            Self::Deferred(dependency) => TextReadyResult::Deferred(dependency),
            Self::Failed(error) => TextReadyResult::Failed(error),
        }
    }
}

enum TextReadyResult<T> {
    Ready(T),
    Deferred(TextMeasureDependency),
    Failed(TextMeasureError),
}

#[cfg(test)]
impl From<TextReadyResult<TextMetrics>> for TextMeasureResult {
    fn from(value: TextReadyResult<TextMetrics>) -> Self {
        match value {
            TextReadyResult::Ready(metrics) => Self::Ready(metrics),
            TextReadyResult::Deferred(dependency) => Self::Deferred(dependency),
            TextReadyResult::Failed(error) => Self::Failed(error),
        }
    }
}

impl From<TextReadyResult<TextLayoutRef>> for TextLayoutResult {
    fn from(value: TextReadyResult<TextLayoutRef>) -> Self {
        match value {
            TextReadyResult::Ready(layout) => Self::Ready(layout),
            TextReadyResult::Deferred(dependency) => Self::Deferred(dependency),
            TextReadyResult::Failed(error) => Self::Failed(error),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextMeasureQuery<'a> {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) node_generation: NodeGeneration,
    pub(crate) text_generation: TextGeneration,
    pub(crate) style_generation: TextGeneration,
    pub(crate) text: &'a str,
    pub(crate) style: &'a ResolvedTextStyle,
    pub(crate) available_inline_width: Option<f32>,
    pub(crate) layout_mode: TextLayoutMode,
    pub(crate) font_generation: FontGeneration,
    pub(crate) scale_generation: u64,
    pub(crate) scale_factor: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureBlocker {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) result: &'static str,
    pub(crate) kind: &'static str,
    pub(crate) reason: Cow<'static, str>,
    pub(crate) duration: Duration,
}

pub(crate) const TEXT_MEASURE_SAMPLE_LIMIT: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureEvent {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) result: &'static str,
    pub(crate) cache: &'static str,
    pub(crate) duration: Duration,
    pub(crate) line_count: Option<usize>,
    pub(crate) min_content_width_bits: Option<u32>,
    pub(crate) max_content_width_bits: Option<u32>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TextMeasureStats {
    pub(crate) query_count: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) measured_count: u64,
    pub(crate) deferred_count: u64,
    pub(crate) failed_count: u64,
    pub(crate) blocked_count: u64,
    pub(crate) total_duration: Duration,
    pub(crate) blockers: Vec<TextMeasureBlocker>,
    pub(crate) events: Vec<TextMeasureEvent>,
}

#[derive(Clone, Debug, PartialEq)]
struct MemoKey {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
    text_generation: TextGeneration,
    style_generation: TextGeneration,
    available_inline_width_bits: Option<u32>,
    layout_mode: TextLayoutMode,
    font_size_bits: u32,
    line_height_policy: TextLineHeightPolicy,
    wrap_policy: TextWrapPolicy,
    white_space_policy: TextWhiteSpacePolicy,
    max_lines: Option<usize>,
    text_overflow: TextOverflow,
    font_generation: FontGeneration,
    scale_generation: u64,
    scale_factor_bits: u32,
}

impl Eq for MemoKey {}

impl Ord for MemoKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.node_id
            .cmp(&other.node_id)
            .then_with(|| self.node_generation.cmp(&other.node_generation))
            .then_with(|| self.text_generation.cmp(&other.text_generation))
            .then_with(|| self.style_generation.cmp(&other.style_generation))
            .then_with(|| {
                self.available_inline_width_bits
                    .cmp(&other.available_inline_width_bits)
            })
            .then_with(|| self.layout_mode.rank().cmp(&other.layout_mode.rank()))
            .then_with(|| self.font_size_bits.cmp(&other.font_size_bits))
            .then_with(|| {
                self.line_height_policy
                    .rank()
                    .cmp(&other.line_height_policy.rank())
            })
            .then_with(|| self.wrap_policy.rank().cmp(&other.wrap_policy.rank()))
            .then_with(|| {
                self.white_space_policy
                    .rank()
                    .cmp(&other.white_space_policy.rank())
            })
            .then_with(|| self.max_lines.cmp(&other.max_lines))
            .then_with(|| {
                text_overflow_rank(self.text_overflow).cmp(&text_overflow_rank(other.text_overflow))
            })
            .then_with(|| self.font_generation.cmp(&other.font_generation))
            .then_with(|| self.scale_generation.cmp(&other.scale_generation))
            .then_with(|| self.scale_factor_bits.cmp(&other.scale_factor_bits))
    }
}

impl PartialOrd for MemoKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub(crate) struct TextMeasureSession<'a> {
    font_manager: &'a FontManager,
    font_generation: FontGeneration,
    memo: BTreeMap<MemoKey, TextLayoutRef>,
    stats: TextMeasureStats,
}

impl<'a> TextMeasureSession<'a> {
    pub(crate) fn new(font_manager: &'a FontManager) -> Self {
        let _ = TextMeasureDependencyKind::ALL;
        let _ = TextMeasureErrorKind::ALL;
        Self {
            font_manager,
            font_generation: font_manager.generation(),
            memo: BTreeMap::new(),
            stats: TextMeasureStats::default(),
        }
    }

    pub(crate) fn font_generation(&self) -> FontGeneration {
        self.font_generation
    }

    pub(crate) fn layout(&mut self, query: TextMeasureQuery<'_>) -> TextLayoutResult {
        TextLayoutResult::from(
            self.measure_inner(query)
                .map_ready(TextMeasureReady::layout),
        )
    }

    #[cfg(test)]
    pub(crate) fn measure(&mut self, query: TextMeasureQuery<'_>) -> TextMeasureResult {
        TextMeasureResult::from(
            self.measure_inner(query)
                .map_ready(|ready| ready.layout.metrics()),
        )
    }

    fn measure_inner(&mut self, query: TextMeasureQuery<'_>) -> TextMeasureOutcome {
        let started = std::time::Instant::now();
        self.stats.query_count += 1;
        if query.font_generation != self.font_manager.generation()
            || query.font_generation != self.font_generation
        {
            let dependency = TextMeasureDependency::new(
                TextMeasureDependencyKind::FontGenerationStale,
                "font generation changed before measurement",
            );
            return self.record_deferred(query.node_id, dependency, started.elapsed());
        }
        if self.font_manager.fallback_snapshot().metadata_count() == 0 {
            let dependency = TextMeasureDependency::new(
                TextMeasureDependencyKind::FontFallbackUnavailable,
                "font fallback snapshot has no entries",
            );
            return self.record_deferred(query.node_id, dependency, started.elapsed());
        }
        if !query.style.font_size().as_px().is_finite() {
            let error = TextMeasureError::new(
                TextMeasureErrorKind::InvalidStyle,
                "font size must be finite for text measurement",
            );
            return self.record_failed(query.node_id, error, started.elapsed());
        }
        if !(query.scale_factor.is_finite() && query.scale_factor > 0.0) {
            let error = TextMeasureError::new(
                TextMeasureErrorKind::InvalidStyle,
                "scale factor must be finite and positive for text measurement",
            );
            return self.record_failed(query.node_id, error, started.elapsed());
        }
        let key = MemoKey::from_query(&query);
        if let Some(layout) = self.memo.get(&key).cloned() {
            let duration = started.elapsed();
            self.stats.cache_hits += 1;
            self.stats.total_duration += duration;
            self.record_event(
                query.node_id,
                "ready",
                "hit",
                duration,
                Some(layout.metrics()),
            );
            return TextMeasureOutcome::Ready(TextMeasureReady { layout });
        }

        self.stats.cache_misses += 1;
        let layout = match layout_uncached(&query, self.font_manager) {
            Ok(layout) => layout,
            Err(error) => return self.record_failed(query.node_id, error, started.elapsed()),
        };
        let metrics = layout.metrics();
        self.memo.insert(key, layout.clone());
        self.stats.measured_count += 1;
        let duration = started.elapsed();
        self.stats.total_duration += duration;
        self.record_event(query.node_id, "ready", "miss", duration, Some(metrics));
        TextMeasureOutcome::Ready(TextMeasureReady { layout })
    }

    pub(crate) fn stats(&self) -> TextMeasureStats {
        self.stats.clone()
    }

    fn record_deferred(
        &mut self,
        node_id: RetainedNodeId,
        dependency: TextMeasureDependency,
        duration: Duration,
    ) -> TextMeasureOutcome {
        self.stats.deferred_count += 1;
        self.stats.blocked_count += 1;
        self.stats.total_duration += duration;
        self.record_blocker(TextMeasureBlocker {
            node_id,
            result: "deferred",
            kind: dependency.kind().as_str(),
            reason: Cow::Owned(dependency.reason().to_owned()),
            duration,
        });
        self.record_event(node_id, "deferred", "none", duration, None);
        TextMeasureOutcome::Deferred(dependency)
    }

    fn record_failed(
        &mut self,
        node_id: RetainedNodeId,
        error: TextMeasureError,
        duration: Duration,
    ) -> TextMeasureOutcome {
        self.stats.failed_count += 1;
        self.stats.blocked_count += 1;
        self.stats.total_duration += duration;
        self.record_blocker(TextMeasureBlocker {
            node_id,
            result: "failed",
            kind: error.kind().as_str(),
            reason: Cow::Owned(error.message().to_owned()),
            duration,
        });
        self.record_event(node_id, "failed", "none", duration, None);
        TextMeasureOutcome::Failed(error)
    }

    fn record_blocker(&mut self, blocker: TextMeasureBlocker) {
        if self.stats.blockers.len() < TEXT_MEASURE_SAMPLE_LIMIT {
            self.stats.blockers.push(blocker);
        }
    }

    fn record_event(
        &mut self,
        node_id: RetainedNodeId,
        result: &'static str,
        cache: &'static str,
        duration: Duration,
        metrics: Option<TextMetrics>,
    ) {
        if self.stats.events.len() >= TEXT_MEASURE_SAMPLE_LIMIT {
            return;
        }
        self.stats.events.push(TextMeasureEvent {
            node_id,
            result,
            cache,
            duration,
            line_count: metrics.map(|metrics| metrics.line_count),
            min_content_width_bits: metrics.map(|metrics| metrics.min_content_width.to_bits()),
            max_content_width_bits: metrics.map(|metrics| metrics.max_content_width.to_bits()),
        });
    }
}

impl MemoKey {
    fn from_query(query: &TextMeasureQuery<'_>) -> Self {
        Self {
            node_id: query.node_id,
            node_generation: query.node_generation,
            text_generation: query.text_generation,
            style_generation: query.style_generation,
            available_inline_width_bits: query.available_inline_width.map(f32::to_bits),
            layout_mode: query.layout_mode,
            font_size_bits: query.style.font_size().as_px().to_bits(),
            line_height_policy: TextLineHeightPolicy::DEFAULT,
            wrap_policy: TextWrapPolicy::DEFAULT,
            white_space_policy: TextWhiteSpacePolicy::DEFAULT,
            max_lines: query.style.max_lines(),
            text_overflow: query.style.text_overflow(),
            font_generation: query.font_generation,
            scale_generation: query.scale_generation,
            scale_factor_bits: query.scale_factor.to_bits(),
        }
    }
}

fn layout_uncached(
    query: &TextMeasureQuery<'_>,
    font_manager: &FontManager,
) -> Result<TextLayoutRef, TextMeasureError> {
    let font_size = query.style.font_size().as_px();
    let line_height = font_size * 1.2;
    let metrics = cosmic_text::Metrics::new(font_size, line_height);
    let buffer = font_manager.with_font_system(|font_system| {
        let mut buffer = cosmic_text::Buffer::new(font_system, metrics);
        buffer.set_wrap(font_system, query.layout_mode.wrap());
        buffer.set_size(
            font_system,
            query.available_inline_width,
            query.layout_mode.height_limit(line_height),
        );
        buffer.set_text(
            font_system,
            query.text,
            &cosmic_text::Attrs::new().weight(cosmic_text::Weight::NORMAL),
            cosmic_text::Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(font_system, true);
        buffer
    });

    let mut glyphs = Vec::new();
    let mut demands = Vec::new();
    let mut demanded_keys = BTreeSet::new();
    let mut width = 0.0_f32;
    let mut max_content_width = 0.0_f32;
    let mut min_content_width = 0.0_f32;
    let mut height = 0.0_f32;
    let mut baseline = 0.0_f32;
    let mut line_count = 0_usize;
    let mut visible_trailing_caret_rect = LayoutRect::ZERO;

    for run in buffer.layout_runs() {
        line_count += 1;
        if line_count == 1 {
            baseline = run.line_y;
        }
        width = width.max(run.line_w);
        max_content_width = max_content_width.max(run.line_w);
        min_content_width = min_content_width.max(min_run_width(run.glyphs));
        height = height.max(run.line_top + run.line_height);
        if query
            .style
            .max_lines()
            .is_none_or(|max_lines| line_count <= max_lines.max(1))
        {
            visible_trailing_caret_rect = LayoutRect::new(
                trailing_run_advance(run.glyphs),
                run.line_top,
                1.0,
                run.line_height.max(1.0),
            );
        }
        for glyph in run.glyphs {
            let physical = glyph.physical((0.0, run.line_y), query.scale_factor);
            let key = GlyphKey::new(physical.cache_key, query.scale_factor);
            glyphs.push(GlyphInstance::new(key, physical.x, physical.y));
            if demanded_keys.insert(key) {
                demands.push(GlyphDemand::new(key));
            }
        }
    }

    let mut line_count = line_count.max(1);
    if let Some(max_lines) = query.style.max_lines() {
        let max_lines = max_lines.max(1);
        if line_count > max_lines {
            line_count = max_lines;
            height = line_height * max_lines as f32;
        }
    }
    if query.text.is_empty() {
        height = line_height;
        baseline = font_size * 0.8;
        visible_trailing_caret_rect = LayoutRect::new(0.0, 0.0, 1.0, line_height.max(1.0));
    } else if visible_trailing_caret_rect == LayoutRect::ZERO {
        visible_trailing_caret_rect = LayoutRect::new(width, 0.0, 1.0, height.max(1.0));
    }

    let layout_key = TextLayoutKey {
        node_id: query.node_id,
        node_generation: query.node_generation,
        text_generation: query.text_generation,
        style_generation: query.style_generation,
        text_hash: stable_text_hash(query.text),
        available_inline_width_bits: query.available_inline_width.map(f32::to_bits),
        layout_mode: query.layout_mode,
        font_size_bits: query.style.font_size().as_px().to_bits(),
        max_lines: query.style.max_lines(),
        text_overflow: query.style.text_overflow(),
        font_generation: query.font_generation,
        scale_generation: query.scale_generation,
        scale_factor_bits: query.scale_factor.to_bits(),
    };
    let generation = TextLayoutGeneration::from_layout_key(&layout_key);
    let metrics = TextMetrics {
        width,
        min_content_width,
        max_content_width,
        height,
        baseline,
        line_count,
    };
    Ok(TextLayoutRef::new(TextLayoutData::new(
        generation,
        layout_key,
        metrics,
        visible_trailing_caret_rect,
        Arc::from(glyphs),
        Arc::from(demands),
    )))
}

fn trailing_run_advance(glyphs: &[cosmic_text::LayoutGlyph]) -> f32 {
    glyphs
        .iter()
        .map(|glyph| glyph.x + glyph.w)
        .fold(0.0_f32, f32::max)
}

fn min_run_width(glyphs: &[cosmic_text::LayoutGlyph]) -> f32 {
    glyphs.iter().map(|glyph| glyph.w).fold(0.0_f32, f32::max)
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

const fn text_overflow_rank(value: TextOverflow) -> u8 {
    match value {
        TextOverflow::Visible => 0,
        TextOverflow::Clip => 1,
        TextOverflow::Ellipsis => 2,
    }
}
