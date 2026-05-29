use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WindowId(u64);

impl WindowId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WindowGeneration(u64);

impl WindowGeneration {
    pub(crate) const INITIAL: Self = Self(1);

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AnyWindowHandle {
    id: WindowId,
    generation: WindowGeneration,
}

impl AnyWindowHandle {
    pub(crate) fn new(id: WindowId, generation: WindowGeneration) -> Self {
        Self { id, generation }
    }

    #[cfg(all(test, target_os = "windows"))]
    pub(crate) fn new_for_tests(id: WindowId) -> Self {
        Self::new(id, WindowGeneration::INITIAL)
    }

    pub fn id(self) -> WindowId {
        self.id
    }

    pub fn generation(self) -> WindowGeneration {
        self.generation
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct WindowHandle<T: 'static> {
    any: AnyWindowHandle,
    marker: PhantomData<fn() -> T>,
}

impl<T: 'static> Clone for WindowHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: 'static> Copy for WindowHandle<T> {}

impl<T: 'static> WindowHandle<T> {
    pub(crate) fn new(any: AnyWindowHandle) -> Self {
        Self {
            any,
            marker: PhantomData,
        }
    }

    pub fn id(self) -> WindowId {
        self.any.id()
    }

    pub fn generation(self) -> WindowGeneration {
        self.any.generation()
    }

    pub fn any(self) -> AnyWindowHandle {
        self.any
    }
}

impl<T: 'static> From<WindowHandle<T>> for AnyWindowHandle {
    fn from(handle: WindowHandle<T>) -> Self {
        handle.any()
    }
}
