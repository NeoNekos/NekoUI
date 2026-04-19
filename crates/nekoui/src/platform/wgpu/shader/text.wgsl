@group(1) @binding(0)
var atlas_sampler: sampler;

@group(1) @binding(1)
var atlas_texture: texture_2d<f32>;

struct MonoTextInstance {
    @location(0) rect: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct ColorTextInstance {
    @location(0) rect: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
    @location(2) alpha: f32,
};

struct MonoVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct ColorVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs_mono(@builtin(vertex_index) vertex_index: u32, instance: MonoTextInstance) -> MonoVsOut {
    let point = instance.rect.xy + UNIT_CORNERS[vertex_index] * instance.rect.zw;

    var out: MonoVsOut;
    out.position = vec4<f32>(rect_to_ndc(point), 0.0, 1.0);
    out.uv = instance.uv_rect.xy + UNIT_CORNERS[vertex_index] * instance.uv_rect.zw;
    out.color = instance.color;
    return out;
}

@fragment
fn fs_mono(in: MonoVsOut) -> @location(0) vec4<f32> {
    let sampled_alpha = textureSample(atlas_texture, atlas_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, sampled_alpha * in.color.a);
}

@vertex
fn vs_color(@builtin(vertex_index) vertex_index: u32, instance: ColorTextInstance) -> ColorVsOut {
    let point = instance.rect.xy + UNIT_CORNERS[vertex_index] * instance.rect.zw;

    var out: ColorVsOut;
    out.position = vec4<f32>(rect_to_ndc(point), 0.0, 1.0);
    out.uv = instance.uv_rect.xy + UNIT_CORNERS[vertex_index] * instance.uv_rect.zw;
    out.alpha = instance.alpha;
    return out;
}

@fragment
fn fs_color(in: ColorVsOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(atlas_texture, atlas_sampler, in.uv);
    return vec4<f32>(sampled.rgb, sampled.a * in.alpha);
}
