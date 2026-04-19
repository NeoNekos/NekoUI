pub(crate) const RECT_SHADER: &str = concat!(
    include_str!("lib/common.wgsl"),
    include_str!("lib/color.wgsl"),
    include_str!("lib/geometry.wgsl"),
    include_str!("lib/sdf.wgsl"),
    include_str!("lib/clip.wgsl"),
    include_str!("lib/gradient.wgsl"),
    include_str!("rect.wgsl"),
);

pub(crate) const TEXT_SHADER: &str = concat!(
    include_str!("lib/common.wgsl"),
    include_str!("lib/color.wgsl"),
    include_str!("lib/geometry.wgsl"),
    include_str!("text.wgsl"),
);
