use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DirtyLaneMask: u8 {
        const BUILD = 0b00001;
        const LAYOUT = 0b00010;
        const PAINT = 0b00100;
        const SEMANTICS = 0b01000;
        const RESOURCES = 0b10000;
    }
}

impl DirtyLaneMask {
    pub fn normalized(self) -> Self {
        if self.contains(Self::BUILD) {
            self | Self::LAYOUT | Self::PAINT
        } else {
            self
        }
    }

    pub fn needs_layout(self) -> bool {
        self.intersects(Self::BUILD | Self::LAYOUT | Self::RESOURCES)
    }

    pub fn needs_scene_compile(self) -> bool {
        self.intersects(Self::BUILD | Self::LAYOUT | Self::PAINT | Self::RESOURCES)
    }
}
