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

    pub fn id(self) -> WindowId {
        self.id
    }

    pub fn generation(self) -> WindowGeneration {
        self.generation
    }
}
