struct RectInstance {
    @location(0) rect: vec4<f32>,
    @location(1) fill_start_color: vec4<f32>,
    @location(2) fill_end_color: vec4<f32>,
    @location(3) fill_meta: vec4<f32>,
    @location(4) corner_radii: vec4<f32>,
    @location(5) border_widths: vec4<f32>,
    @location(6) border_color: vec4<f32>,
};

struct RectVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) fill_start_color: vec4<f32>,
    @location(3) fill_end_color: vec4<f32>,
    @location(4) fill_meta: vec4<f32>,
    @location(5) corner_radii: vec4<f32>,
    @location(6) border_widths: vec4<f32>,
    @location(7) border_color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: RectInstance,
) -> RectVsOut {
    let point = instance.rect.xy + UNIT_CORNERS[vertex_index] * instance.rect.zw;

    var out: RectVsOut;
    out.position = vec4<f32>(rect_to_ndc(point), 0.0, 1.0);
    out.local_pos = UNIT_CORNERS[vertex_index] * instance.rect.zw;
    out.size = instance.rect.zw;
    out.fill_start_color = instance.fill_start_color;
    out.fill_end_color = instance.fill_end_color;
    out.fill_meta = instance.fill_meta;
    out.corner_radii = instance.corner_radii;
    out.border_widths = instance.border_widths;
    out.border_color = instance.border_color;
    return out;
}

fn sample_fill(in: RectVsOut) -> vec4<f32> {
    if in.fill_meta.x < 0.5 {
        return in.fill_start_color;
    }

    let angle = in.fill_meta.y;
    let dir = vec2<f32>(cos(angle), sin(angle));
    let centered = in.local_pos - in.size * 0.5;
    let extent = max(abs(dir.x) * in.size.x * 0.5 + abs(dir.y) * in.size.y * 0.5, 0.0001);
    let projection = dot(centered, dir);
    let t = 0.5 + projection / (extent * 2.0);
    return sample_linear_gradient(in.fill_start_color, in.fill_end_color, t);
}

fn corner_radius_for(local_pos: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let is_left = local_pos.x < size.x * 0.5;
    let is_top = local_pos.y < size.y * 0.5;
    if is_top {
        if is_left {
            return radii.x;
        }
        return radii.y;
    }
    if is_left {
        return radii.w;
    }
    return radii.z;
}

fn rect_sdf(local_pos: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let radius = min(corner_radius_for(local_pos, size, radii), 0.5 * min(size.x, size.y));
    let half_size = size * 0.5;
    let centered = local_pos - half_size;
    let q = abs(centered) - (half_size - vec2<f32>(radius));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn inner_corner_radii(radii: vec4<f32>, border_widths: vec4<f32>) -> vec4<f32> {
    let top = border_widths.x;
    let right = border_widths.y;
    let bottom = border_widths.z;
    let left = border_widths.w;
    return max(
        radii - vec4<f32>(
            max(left, top),
            max(right, top),
            max(right, bottom),
            max(left, bottom),
        ),
        vec4<f32>(0.0),
    );
}

@fragment
fn fs_main(in: RectVsOut) -> @location(0) vec4<f32> {
    let outer_sdf = rect_sdf(in.local_pos, in.size, in.corner_radii);
    let aa = max(fwidth(outer_sdf), 1.0);
    let outer_alpha = 1.0 - smoothstep(0.0, aa, outer_sdf);
    let fill_color = sample_fill(in);

    let has_border = any(in.border_widths > vec4<f32>(0.0)) && in.border_color.a > 0.0;
    if !has_border {
        return vec4<f32>(fill_color.rgb, fill_color.a * outer_alpha);
    }

    let inner_origin = vec2<f32>(in.border_widths.w, in.border_widths.x);
    let inner_size = vec2<f32>(
        max(in.size.x - (in.border_widths.w + in.border_widths.y), 0.0),
        max(in.size.y - (in.border_widths.x + in.border_widths.z), 0.0),
    );

    if inner_size.x <= 0.0 || inner_size.y <= 0.0 {
        return vec4<f32>(in.border_color.rgb, in.border_color.a * outer_alpha);
    }

    let inner_sdf = rect_sdf(
        in.local_pos - inner_origin,
        inner_size,
        inner_corner_radii(in.corner_radii, in.border_widths),
    );
    let inner_alpha = 1.0 - smoothstep(0.0, aa, inner_sdf);
    let color = mix(in.border_color, fill_color, inner_alpha);
    return vec4<f32>(color.rgb, color.a * outer_alpha);
}
