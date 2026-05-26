use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Weak;

use crate::error::{NekoError, NekoResult};
use crate::runtime::Runtime;
use crate::runtime::entity_store::EntityKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct SubscriptionId(u64);

impl SubscriptionId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct SubscriptionGeneration(u64);

impl SubscriptionGeneration {
    pub(crate) const INITIAL: Self = Self(1);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct SubscriptionKey {
    id: SubscriptionId,
    generation: SubscriptionGeneration,
}

impl SubscriptionKey {
    pub(crate) fn new(id: SubscriptionId, generation: SubscriptionGeneration) -> Self {
        Self { id, generation }
    }

    pub(crate) fn id(self) -> SubscriptionId {
        self.id
    }

    pub(crate) fn generation(self) -> SubscriptionGeneration {
        self.generation
    }
}

pub(crate) type SubscriptionCallback =
    Box<dyn FnMut(&mut dyn Any, &mut Runtime, EntityKey) -> NekoResult<()>>;

pub(crate) struct SubscriptionRecord {
    generation: SubscriptionGeneration,
    source: EntityKey,
    target: EntityKey,
    owner: Weak<()>,
    callback: Option<SubscriptionCallback>,
}

impl SubscriptionRecord {
    fn new(
        generation: SubscriptionGeneration,
        source: EntityKey,
        target: EntityKey,
        owner: Weak<()>,
        callback: SubscriptionCallback,
    ) -> Self {
        Self {
            generation,
            source,
            target,
            owner,
            callback: Some(callback),
        }
    }

    fn is_live(&self, key: SubscriptionKey) -> bool {
        self.generation == key.generation() && self.owner.upgrade().is_some()
    }
}

#[derive(Default)]
pub(crate) struct SubscriptionStore {
    next_id: u64,
    records: BTreeMap<SubscriptionId, SubscriptionRecord>,
}

impl fmt::Debug for SubscriptionStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubscriptionStore")
            .field("next_id", &self.next_id)
            .field("records_len", &self.records.len())
            .finish()
    }
}

impl SubscriptionStore {
    pub(crate) fn insert(
        &mut self,
        source: EntityKey,
        target: EntityKey,
        owner: Weak<()>,
        callback: SubscriptionCallback,
    ) -> SubscriptionKey {
        self.next_id += 1;
        let key = SubscriptionKey::new(
            SubscriptionId::new(self.next_id),
            SubscriptionGeneration::INITIAL,
        );
        self.records.insert(
            key.id(),
            SubscriptionRecord::new(key.generation(), source, target, owner, callback),
        );
        key
    }

    pub(crate) fn live_for_source(&self, source: EntityKey) -> (Vec<SubscriptionKey>, u64) {
        let mut live = Vec::new();
        let mut cancelled = 0;
        for (id, record) in &self.records {
            if record.source != source {
                continue;
            }
            let key = SubscriptionKey::new(*id, record.generation);
            if record.is_live(key) {
                live.push(key);
            } else {
                cancelled += 1;
            }
        }
        (live, cancelled)
    }

    pub(crate) fn take_callback(
        &mut self,
        key: SubscriptionKey,
    ) -> NekoResult<(EntityKey, SubscriptionCallback)> {
        let record = self
            .records
            .get_mut(&key.id())
            .ok_or_else(|| NekoError::stale("subscription id is not registered"))?;
        if !record.is_live(key) {
            return Err(NekoError::stale("subscription is cancelled"));
        }
        let callback = record
            .callback
            .take()
            .ok_or_else(|| NekoError::invalid_input("subscription callback is already running"))?;
        Ok((record.target, callback))
    }

    pub(crate) fn restore_callback(
        &mut self,
        key: SubscriptionKey,
        callback: SubscriptionCallback,
    ) -> NekoResult<()> {
        let record = self
            .records
            .get_mut(&key.id())
            .ok_or_else(|| NekoError::stale("subscription id is not registered"))?;
        if !record.is_live(key) {
            return Err(NekoError::stale("subscription is cancelled"));
        }
        record.callback = Some(callback);
        Ok(())
    }
}
