use std::collections::BTreeMap;

use crate::error::{NekoError, NekoResult};
use crate::retained::{DirtyCause, IdentitySeed, RetainedDirty, RetainedIdentity};
use crate::runtime::scheduler::Scheduler;
use crate::window::{AnyWindowHandle, WindowGeneration, WindowId, WindowOptions, WindowRecord};

#[derive(Debug)]
pub struct RuntimeState {
    next_window_id: u64,
    windows: BTreeMap<WindowId, WindowRecord>,
    scheduler: Scheduler,
    retained_seed: IdentitySeed,
    retained_dirty: Vec<RetainedDirty>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            next_window_id: 1,
            windows: BTreeMap::new(),
            scheduler: Scheduler::default(),
            retained_seed: IdentitySeed::default(),
            retained_dirty: Vec::new(),
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
        self.windows
            .insert(handle.id(), WindowRecord::new(handle, options));
        Ok(())
    }

    pub fn request_close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.request_close();
        Ok(())
    }

    pub fn close_window(&mut self, handle: AnyWindowHandle) -> NekoResult<()> {
        let window = self.window_mut(handle)?;
        window.close();
        Ok(())
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

    pub fn window_mut(&mut self, handle: AnyWindowHandle) -> NekoResult<&mut WindowRecord> {
        let window = self
            .windows
            .get_mut(&handle.id())
            .ok_or_else(|| NekoError::stale("window id is not registered"))?;
        window.ensure_live(handle)?;
        Ok(window)
    }

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

    pub fn live_window_count(&self) -> usize {
        self.live_window_ids().len()
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    pub fn emit_retained_dirty(&mut self, identity: Option<RetainedIdentity>, cause: DirtyCause) {
        let identity = identity.or_else(|| Some(self.retained_seed.allocate()));
        self.retained_dirty.push(RetainedDirty { identity, cause });
    }

    #[cfg(test)]
    pub fn retained_dirty(&self) -> &[RetainedDirty] {
        &self.retained_dirty
    }
}
