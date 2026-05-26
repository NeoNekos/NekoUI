use std::marker::PhantomData;
use std::rc::{Rc, Weak};

use crate::app::{AppContext, Context, Subscription};
use crate::error::{NekoError, NekoResult};
use crate::runtime::entity_store::EntityKey;

pub trait EntityAccess {
    #[doc(hidden)]
    fn __entity_read<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        read: impl FnOnce(&T) -> R,
    ) -> NekoResult<R>;

    #[doc(hidden)]
    fn __entity_update<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R>;

    #[doc(hidden)]
    fn __entity_stale(&mut self);
}

#[derive(Debug)]
pub struct Entity<T: 'static> {
    pub(crate) key: EntityKey,
    pub(crate) owner: Rc<()>,
    marker: PhantomData<fn() -> T>,
}

impl<T: 'static> Clone for Entity<T> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            owner: Rc::clone(&self.owner),
            marker: PhantomData,
        }
    }
}

impl<T: 'static> Entity<T> {
    pub(crate) fn new(key: EntityKey, owner: Rc<()>) -> Self {
        Self {
            key,
            owner,
            marker: PhantomData,
        }
    }

    pub(crate) fn key(&self) -> EntityKey {
        self.key
    }

    pub fn read<C, R>(&self, cx: &mut C, read: impl FnOnce(&T) -> R) -> NekoResult<R>
    where
        C: EntityAccess,
    {
        cx.__entity_read(self, read)
    }

    pub fn update<C, R>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R>
    where
        C: EntityAccess,
    {
        cx.__entity_update(self, update)
    }

    pub fn downgrade(&self) -> WeakEntity<T> {
        WeakEntity {
            key: self.key,
            owner: Rc::downgrade(&self.owner),
            marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct WeakEntity<T: 'static> {
    pub(crate) key: EntityKey,
    pub(crate) owner: Weak<()>,
    marker: PhantomData<fn() -> T>,
}

impl<T: 'static> Clone for WeakEntity<T> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            owner: self.owner.clone(),
            marker: PhantomData,
        }
    }
}

impl<T: 'static> WeakEntity<T> {
    pub fn upgrade<C: EntityAccess>(&self, cx: &mut C) -> NekoResult<Entity<T>> {
        let owner = self.owner.upgrade().ok_or_else(|| {
            cx.__entity_stale();
            NekoError::stale("entity handle is stale")
        })?;
        let entity = Entity::new(self.key, owner);
        entity.read(cx, |_| ())?;
        Ok(entity)
    }

    pub fn update<C, R>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R>
    where
        C: EntityAccess,
    {
        self.upgrade(cx)?.update(cx, update)
    }
}

impl EntityAccess for AppContext<'_> {
    fn __entity_read<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        read: impl FnOnce(&T) -> R,
    ) -> NekoResult<R> {
        self.runtime().read_entity(entity, read)
    }

    fn __entity_update<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R> {
        self.runtime().update_entity(entity, update)
    }

    fn __entity_stale(&mut self) {
        self.runtime().record_api_stale();
    }
}

impl<U: 'static> EntityAccess for Context<'_, U> {
    fn __entity_read<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        read: impl FnOnce(&T) -> R,
    ) -> NekoResult<R> {
        self.runtime().read_entity(entity, read)
    }

    fn __entity_update<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<'_, T>) -> NekoResult<R>,
    ) -> NekoResult<R> {
        self.runtime().update_entity(entity, update)
    }

    fn __entity_stale(&mut self) {
        self.runtime().record_api_stale();
    }
}

impl<T: 'static> Context<'_, T> {
    pub fn observe<U: 'static>(
        &mut self,
        source: &Entity<U>,
        callback: impl FnMut(&mut T, Entity<U>, &mut Context<'_, T>) + 'static,
    ) -> Subscription {
        let entity = self.entity_key();
        self.runtime().observe_entity(entity, source, callback)
    }
}
