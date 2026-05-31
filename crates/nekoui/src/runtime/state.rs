use std::collections::BTreeMap;

use crate::error::{NekoError, NekoResult};
use crate::interaction::InteractionState;
use crate::layout::{LayoutPassStats, LayoutRect, LayoutSize, LayoutTreeSnapshot};
use crate::platform::{ImePlatformRequest, PhysicalSize, Renderability};
use crate::render::{FrameGraphStats, PreparedFrame};
use crate::retained::{
    DirtyCause, RetainedDiffStats, RetainedDirty, RetainedIdentity, RetainedTree,
    RetainedTreeSnapshot,
};
use crate::runtime::entity_store::EntityStore;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::subscription_store::SubscriptionStore;
use crate::scene::{PaintScene, SceneCompileStats};
use crate::semantics::{SemanticBuildStats, SemanticTreeSnapshot};
use crate::text::FontManager;
use crate::window::{
    AnyWindowHandle, WindowGeneration, WindowId, WindowLifecycle, WindowOptions, WindowRecord,
};

#[derive(Debug)]
pub struct RuntimeState {
    next_window_id: u64,
    windows: BTreeMap<WindowId, WindowRecord>,
    scheduler: Scheduler,
    retained_trees: BTreeMap<WindowId, RetainedTree>,
    layout_snapshots: BTreeMap<WindowId, LayoutTreeSnapshot>,
    semantic_snapshots: BTreeMap<WindowId, SemanticTreeSnapshot>,
    scene_snapshots: BTreeMap<WindowId, PaintScene>,
    prepared_frame_snapshots: BTreeMap<WindowId, PreparedFrame>,
    retained_dirty: Vec<RetainedDirty>,
    last_retained_diff: RetainedDiffStats,
    consumed_dirty_lanes: BTreeMap<WindowId, crate::diagnostic::DirtyLanes>,
    last_layout_pass: LayoutPassStats,
    last_semantic_build: SemanticBuildStats,
    last_scene_compile: SceneCompileStats,
    last_frame_graph: FrameGraphStats,
    font_manager: FontManager,
    entity_store: EntityStore,
    subscription_store: SubscriptionStore,
    interaction: BTreeMap<WindowId, InteractionState>,
    ime_requests: BTreeMap<WindowId, Vec<ImePlatformRequest>>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            next_window_id: 1,
            windows: BTreeMap::new(),
            scheduler: Scheduler::default(),
            retained_trees: BTreeMap::new(),
            layout_snapshots: BTreeMap::new(),
            semantic_snapshots: BTreeMap::new(),
            scene_snapshots: BTreeMap::new(),
            prepared_frame_snapshots: BTreeMap::new(),
            retained_dirty: Vec::new(),
            last_retained_diff: RetainedDiffStats::default(),
            consumed_dirty_lanes: BTreeMap::new(),
            last_layout_pass: LayoutPassStats::default(),
            last_semantic_build: SemanticBuildStats::default(),
            last_scene_compile: SceneCompileStats::default(),
            last_frame_graph: FrameGraphStats::default(),
            font_manager: FontManager::default(),
            entity_store: EntityStore::default(),
            subscription_store: SubscriptionStore::default(),
            interaction: BTreeMap::new(),
            ime_requests: BTreeMap::new(),
        }
    }
}

impl RuntimeState {
    pub fn allocate_window_handle(&mut self) -> AnyWindowHandle {
        let id = WindowId::new(self.next_window_id);
        self.next_window_id += 1;
        AnyWindowHandle::new(id, WindowGeneration::INITIAL)
    }

    pub fn open_window(
        &mut self,
        handle: AnyWindowHandle,
        options: WindowOptions,
    ) -> NekoResult<()> {
        if self.windows.contains_key(&handle.id()) {
            return Err(NekoError::invalid_input("window id already exists"));
        }

        self.scheduler.ensure_window(handle.id());
        self.retained_trees.entry(handle.id()).or_default();
        self.interaction.entry(handle.id()).or_default();
        self.layout_snapshots.remove(&handle.id());
        self.semantic_snapshots.remove(&handle.id());
        self.scene_snapshots.remove(&handle.id());
        self.prepared_frame_snapshots.remove(&handle.id());
        self.windows
            .insert(handle.id(), WindowRecord::new(handle, options));
        Ok(())
    }

    pub fn request_close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.request_close();
        Ok(())
    }

    pub fn confirm_close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.confirm_close();
        self.scheduler
            .set_renderability(handle.id(), Renderability::Closing);
        Ok(())
    }

    pub fn close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.close();
        self.scheduler
            .set_renderability(handle.id(), Renderability::Destroyed);
        self.retained_trees.remove(&handle.id());
        self.layout_snapshots.remove(&handle.id());
        self.semantic_snapshots.remove(&handle.id());
        self.scene_snapshots.remove(&handle.id());
        self.prepared_frame_snapshots.remove(&handle.id());
        self.interaction.remove(&handle.id());
        self.ime_requests.remove(&handle.id());
        Ok(())
    }

    pub fn resize_window(
        &mut self,
        handle: AnyWindowHandle,
        logical_size: LayoutSize,
    ) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.resize(logical_size);
        Ok(())
    }

    pub fn rescale_window(
        &mut self,
        handle: AnyWindowHandle,
        scale_factor: f32,
    ) -> NekoResult<bool> {
        let window = self.window_mut(handle)?;
        window.rescale(scale_factor)
    }

    pub fn mark_native_window_created(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.mark_native_created();
        self.scheduler.ensure_window(handle.id());
        Ok(())
    }

    pub fn show_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.show();
        Ok(())
    }

    pub fn minimize_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.minimize();
        self.scheduler
            .set_renderability(handle.id(), Renderability::Minimized);
        Ok(())
    }

    pub fn restore_window(&mut self, handle: AnyWindowHandle) -> NekoResult<bool> {
        let renderability = {
            let window = self.window_mut(handle)?;
            window.restore();
            window.renderability()
        };
        let outcome = self.scheduler.set_renderability(handle.id(), renderability);
        Ok(matches!(
            outcome,
            crate::runtime::scheduler::RedrawRequestOutcome::Requested
        ))
    }

    pub fn set_window_physical_size(
        &mut self,
        handle: AnyWindowHandle,
        physical_size: PhysicalSize,
    ) -> NekoResult<bool> {
        let window = self.window_mut(handle)?;
        Ok(window.set_physical_size(physical_size))
    }

    pub fn set_window_renderability(
        &mut self,
        handle: AnyWindowHandle,
        renderability: Renderability,
    ) -> NekoResult<crate::runtime::scheduler::RedrawRequestOutcome> {
        let window = self.window_mut(handle)?;
        window.set_renderability(renderability);
        Ok(self.scheduler.set_renderability(handle.id(), renderability))
    }

    pub fn validate_window(&self, handle: AnyWindowHandle) -> NekoResult<()> {
        self.window(handle).map(|_| ())
    }

    pub fn window(&self, handle: AnyWindowHandle) -> NekoResult<&WindowRecord> {
        let window = self
            .windows
            .get(&handle.id())
            .ok_or_else(|| NekoError::stale("window id is not registered"))?;
        window.ensure_live(handle)?;
        Ok(window)
    }

    pub fn window_by_id(&self, id: WindowId) -> NekoResult<&WindowRecord> {
        self.windows
            .get(&id)
            .ok_or_else(|| NekoError::stale("window id is not registered"))
            .and_then(|window| {
                window.ensure_live(window.handle())?;
                Ok(window)
            })
    }

    pub fn window_mut(&mut self, handle: AnyWindowHandle) -> NekoResult<&mut WindowRecord> {
        let window = self
            .windows
            .get_mut(&handle.id())
            .ok_or_else(|| NekoError::stale("window id is not registered"))?;
        window.ensure_live(handle)?;
        Ok(window)
    }

    #[cfg(test)]
    pub fn windows(&self) -> impl Iterator<Item = &WindowRecord> {
        self.windows.values()
    }

    pub fn live_window_ids(&self) -> Vec<WindowId> {
        self.windows
            .values()
            .filter(|window| self.validate_window(window.handle()).is_ok())
            .map(|window| window.handle().id())
            .collect()
    }

    pub(crate) fn windows_needing_native_creation(&self) -> Vec<WindowRecord> {
        self.windows
            .values()
            .filter(|window| {
                self.validate_window(window.handle()).is_ok() && !window.native_created()
            })
            .cloned()
            .collect()
    }

    pub(crate) fn live_windows(&self) -> Vec<WindowRecord> {
        self.windows
            .values()
            .filter(|window| self.validate_window(window.handle()).is_ok())
            .cloned()
            .collect()
    }

    pub(crate) fn closing_windows_for_platform(&self) -> Vec<AnyWindowHandle> {
        self.windows
            .values()
            .filter(|window| window.lifecycle() == WindowLifecycle::Closing)
            .map(WindowRecord::handle)
            .collect()
    }
    pub fn live_window_count(&self) -> usize {
        self.live_window_ids().len()
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    pub fn font_manager(&self) -> &FontManager {
        &self.font_manager
    }

    pub(crate) fn entity_store(&self) -> &EntityStore {
        &self.entity_store
    }

    pub(crate) fn entity_store_mut(&mut self) -> &mut EntityStore {
        &mut self.entity_store
    }

    pub(crate) fn subscription_store(&self) -> &SubscriptionStore {
        &self.subscription_store
    }

    pub(crate) fn subscription_store_mut(&mut self) -> &mut SubscriptionStore {
        &mut self.subscription_store
    }

    #[cfg(test)]
    pub(crate) fn font_manager_mut(&mut self) -> &mut FontManager {
        &mut self.font_manager
    }

    pub fn retained_tree_mut(&mut self, window: WindowId) -> Option<&mut RetainedTree> {
        self.retained_trees.get_mut(&window)
    }

    pub fn retained_tree(&self, window: WindowId) -> Option<&RetainedTree> {
        self.retained_trees.get(&window)
    }

    pub fn retained_snapshot(&self, window: WindowId) -> Option<RetainedTreeSnapshot> {
        self.retained_trees.get(&window).map(RetainedTree::snapshot)
    }

    pub fn style_snapshot(&self, window: WindowId) -> Option<crate::style::StyleTreeSnapshot> {
        self.retained_trees
            .get(&window)
            .map(RetainedTree::style_snapshot)
    }

    pub fn layout_snapshot(&self, window: WindowId) -> Option<LayoutTreeSnapshot> {
        self.layout_snapshots.get(&window).cloned()
    }

    #[cfg(test)]
    pub fn semantic_snapshot(&self, window: WindowId) -> Option<SemanticTreeSnapshot> {
        self.semantic_snapshots.get(&window).cloned()
    }

    pub fn scene_snapshot(&self, window: WindowId) -> Option<PaintScene> {
        self.scene_snapshots.get(&window).cloned()
    }

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn prepared_frame_snapshot(&self, window: WindowId) -> Option<PreparedFrame> {
        self.prepared_frame_snapshots.get(&window).cloned()
    }

    pub(crate) fn interaction(&self, window: WindowId) -> Option<&InteractionState> {
        self.interaction.get(&window)
    }

    pub(crate) fn interaction_mut(&mut self, window: WindowId) -> &mut InteractionState {
        self.interaction.entry(window).or_default()
    }

    pub(crate) fn push_ime_request(&mut self, window: WindowId, request: ImePlatformRequest) {
        self.ime_requests.entry(window).or_default().push(request);
    }

    pub(crate) fn take_ime_requests(&mut self, window: WindowId) -> Vec<ImePlatformRequest> {
        self.ime_requests.remove(&window).unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn peek_ime_requests(&self, window: WindowId) -> &[ImePlatformRequest] {
        self.ime_requests.get(&window).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn replace_ime_candidate_rect(&mut self, window: WindowId, rect: LayoutRect) {
        let requests = self.ime_requests.entry(window).or_default();
        requests.retain(|request| !matches!(request, ImePlatformRequest::CursorArea { .. }));
        requests.push(ImePlatformRequest::CursorArea { rect });
    }

    pub fn layout_node_count(&self) -> usize {
        self.layout_snapshots
            .values()
            .map(LayoutTreeSnapshot::node_count)
            .sum()
    }

    pub fn set_layout_snapshot(&mut self, window: WindowId, snapshot: LayoutTreeSnapshot) {
        self.layout_snapshots.insert(window, snapshot);
    }

    pub fn set_semantic_snapshot(&mut self, window: WindowId, snapshot: SemanticTreeSnapshot) {
        self.semantic_snapshots.insert(window, snapshot);
    }

    pub fn set_scene_snapshot(&mut self, window: WindowId, snapshot: PaintScene) {
        self.scene_snapshots.insert(window, snapshot);
    }

    pub(crate) fn set_prepared_frame_snapshot(
        &mut self,
        window: WindowId,
        snapshot: PreparedFrame,
    ) {
        self.prepared_frame_snapshots.insert(window, snapshot);
    }

    pub(crate) fn prepared_frame_count(&self) -> usize {
        self.prepared_frame_snapshots.len()
    }

    pub fn scene_node_count(&self) -> usize {
        self.scene_snapshots
            .values()
            .map(|scene| scene.stats().node_count)
            .sum()
    }

    pub fn set_last_layout_pass(&mut self, stats: LayoutPassStats) {
        self.last_layout_pass = stats;
    }

    pub fn last_layout_pass(&self) -> &LayoutPassStats {
        &self.last_layout_pass
    }

    pub fn set_last_semantic_build(&mut self, stats: SemanticBuildStats) {
        self.last_semantic_build = stats;
    }

    pub fn set_last_scene_compile(&mut self, stats: SceneCompileStats) {
        self.last_scene_compile = stats;
    }

    pub fn last_scene_compile(&self) -> &SceneCompileStats {
        &self.last_scene_compile
    }

    pub(crate) fn set_last_frame_graph(&mut self, stats: FrameGraphStats) {
        self.last_frame_graph = stats;
    }

    pub(crate) fn last_frame_graph(&self) -> &FrameGraphStats {
        &self.last_frame_graph
    }

    pub fn record_consumed_dirty_lanes(
        &mut self,
        window: WindowId,
        lanes: crate::diagnostic::DirtyLanes,
    ) {
        self.consumed_dirty_lanes
            .entry(window)
            .or_default()
            .insert(lanes);
    }

    #[cfg(test)]
    pub fn clear_consumed_dirty_lanes(&mut self, window: WindowId) {
        self.consumed_dirty_lanes.remove(&window);
    }

    pub fn reported_dirty_lanes(&self, window: WindowId) -> crate::diagnostic::DirtyLanes {
        let current = self
            .scheduler
            .window_states()
            .get(&window)
            .map_or(crate::diagnostic::DirtyLanes::empty(), |state| {
                state.dirty_lanes()
            });
        current
            | self
                .consumed_dirty_lanes
                .get(&window)
                .copied()
                .unwrap_or_default()
    }

    pub fn retained_node_count(&self) -> usize {
        self.retained_trees
            .values()
            .map(|tree| tree.snapshot().node_count())
            .sum()
    }

    pub fn set_last_retained_diff(&mut self, stats: RetainedDiffStats) {
        self.last_retained_diff = stats;
    }

    pub fn last_retained_diff(&self) -> &RetainedDiffStats {
        &self.last_retained_diff
    }

    pub fn emit_retained_dirty(
        &mut self,
        identity: Option<RetainedIdentity>,
        cause: DirtyCause,
        lanes: crate::diagnostic::DirtyLanes,
    ) {
        self.retained_dirty
            .push(RetainedDirty::new(identity, cause, lanes));
    }

    pub fn extend_retained_dirty(&mut self, dirty: impl IntoIterator<Item = RetainedDirty>) {
        self.retained_dirty.extend(dirty);
    }

    #[cfg(test)]
    pub fn retained_dirty(&self) -> &[RetainedDirty] {
        &self.retained_dirty
    }
}
