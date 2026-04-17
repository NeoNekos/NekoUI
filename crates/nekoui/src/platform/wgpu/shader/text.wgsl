struct ViewUniForm {
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> view: ViewUniForm;

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

fn rect_to_ndc(point: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (point.x / view.viewport.x) * 2.0 - 1.0,
        1.0 - (point.y / view.viewport.y) * 2.0,
    );
}

@vertex
fn vs_mono(@builtin(vertex_index) vertex_index: u32, instance: MonoTextInstance) -> MonoVsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    let point = instance.rect.xy + corners[vertex_index] * instance.rect.zw;

    var out: MonoVsOut;
    out.position = vec4<f32>(rect_to_ndc(point), 0.0, 1.0);
    out.uv = instance.uv_rect.xy + corners[vertex_index] * instance.uv_rect.zw;
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
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    let point = instance.rect.xy + corners[vertex_index] * instance.rect.zw;

    var out: ColorVsOut;
    out.position = vec4<f32>(rect_to_ndc(point), 0.0, 1.0);
    out.uv = instance.uv_rect.xy + corners[vertex_index] * instance.uv_rect.zw;
    out.alpha = instance.alpha;
    return out;
}

@fragment
fn fs_color(in: ColorVsOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(atlas_texture, atlas_sampler, in.uv);
    return vec4<f32>(sampled.rgb, sampled.a * in.alpha);
}
