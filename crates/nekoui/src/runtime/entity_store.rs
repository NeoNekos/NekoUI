use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Weak;

use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct EntityId(u64);

impl EntityId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[cfg(test)]
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct EntityGeneration(u64);

impl EntityGeneration {
    pub(crate) const INITIAL: Self = Self(1);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct EntityKey {
    id: EntityId,
    generation: EntityGeneration,
}

impl EntityKey {
    pub(crate) fn new(id: EntityId, generation: EntityGeneration) -> Self {
        Self { id, generation }
    }

    pub(crate) fn id(self) -> EntityId {
        self.id
    }

    pub(crate) fn generation(self) -> EntityGeneration {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn raw_id(self) -> u64 {
        self.id.raw()
    }
}

struct EntityRecord {
    generation: EntityGeneration,
    owner: Weak<()>,
    value: Option<Box<dyn Any>>,
}

impl EntityRecord {
    fn new<T: 'static>(generation: EntityGeneration, owner: Weak<()>, value: T) -> Self {
        Self {
            generation,
            owner,
            value: Some(Box::new(value)),
        }
    }

    fn validate(&self, key: EntityKey) -> NekoResult<()> {
        if self.generation != key.generation() || self.owner.upgrade().is_none() {
            return Err(NekoError::stale("entity handle is stale"));
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct EntityStore {
    next_id: u64,
    records: BTreeMap<EntityId, EntityRecord>,
}

impl fmt::Debug for EntityStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EntityStore")
            .field("next_id", &self.next_id)
            .field("records_len", &self.records.len())
            .finish()
    }
}

impl EntityStore {
    pub(crate) fn reserve(&mut self) -> EntityKey {
        self.next_id += 1;
        EntityKey::new(EntityId::new(self.next_id), EntityGeneration::INITIAL)
    }

    pub(crate) fn insert_reserved<T: 'static>(
        &mut self,
        key: EntityKey,
        value: T,
        owner: Weak<()>,
    ) {
        self.records
            .insert(key.id(), EntityRecord::new(key.generation(), owner, value));
    }

    pub(crate) fn read<T: 'static, R>(
        &self,
        key: EntityKey,
        read: impl FnOnce(&T) -> R,
    ) -> NekoResult<R> {
        let record = self
            .records
            .get(&key.id())
            .ok_or_else(|| NekoError::stale("entity id is not registered"))?;
        record.validate(key)?;
        let value = record
            .value
            .as_deref()
            .ok_or_else(|| NekoError::invalid_input("entity is already updating"))?
            .downcast_ref::<T>()
            .ok_or_else(|| NekoError::invalid_input("entity type does not match handle"))?;
        Ok(read(value))
    }

    pub(crate) fn take_any(&mut self, key: EntityKey) -> NekoResult<Box<dyn Any>> {
        let record = self
            .records
            .get_mut(&key.id())
            .ok_or_else(|| NekoError::stale("entity id is not registered"))?;
        record.validate(key)?;
        record
            .value
            .take()
            .ok_or_else(|| NekoError::invalid_input("entity is already updating"))
    }

    pub(crate) fn restore_any(&mut self, key: EntityKey, value: Box<dyn Any>) -> NekoResult<()> {
        let record = self
            .records
            .get_mut(&key.id())
            .ok_or_else(|| NekoError::stale("entity id is not registered"))?;
        record.validate(key)?;
        if record.value.is_some() {
            return Err(NekoError::invalid_input("entity value is already present"));
        }
        record.value = Some(value);
        Ok(())
    }
}
