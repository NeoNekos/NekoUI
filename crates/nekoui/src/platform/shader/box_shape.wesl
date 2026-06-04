struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) local_position: vec2<f32>,
    @location(2) size: vec4<f32>,
    @location(3) corner_radii: vec4<f32>,
    @location(4) border_widths: vec4<f32>,
    @location(5) fill_color: vec4<f32>,
    @location(6) border_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_position: vec2<f32>,
    @location(1) size: vec4<f32>,
    @location(2) corner_radii: vec4<f32>,
    @location(3) border_widths: vec4<f32>,
    @location(4) fill_color: vec4<f32>,
    @location(5) border_color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.local_position = input.local_position;
    output.size = input.size;
    output.corner_radii = input.corner_radii;
    output.border_widths = input.border_widths;
    output.fill_color = input.fill_color;
    output.border_color = input.border_color;
    return output;
}

fn radius_scale_for(size: f32, radius_sum: f32) -> f32 {
    return select(1.0, min(1.0, size / max(radius_sum, 0.000001)), radius_sum > 0.0);
}

fn normalized_corner_radii(size: vec2<f32>, radii: vec4<f32>) -> vec4<f32> {
    let positive = max(radii, vec4<f32>(0.0));
    let top_scale = radius_scale_for(size.x, positive.x + positive.y);
    let bottom_scale = radius_scale_for(size.x, positive.w + positive.z);
    let left_scale = radius_scale_for(size.y, positive.x + positive.w);
    let right_scale = radius_scale_for(size.y, positive.y + positive.z);
    let scale = min(min(top_scale, bottom_scale), min(left_scale, right_scale));
    return positive * scale;
}

fn corner_radius_for(centered: vec2<f32>, radii: vec4<f32>) -> f32 {
    let top_radius = select(radii.y, radii.x, centered.x < 0.0);
    let bottom_radius = select(radii.z, radii.w, centered.x < 0.0);
    return select(bottom_radius, top_radius, centered.y < 0.0);
}

fn rounded_box_distance(point: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let safe_size = max(size, vec2<f32>(0.0));
    let normalized_radii = normalized_corner_radii(safe_size, radii);
    let center = safe_size * 0.5;
    let centered = point - center;
    let radius = corner_radius_for(centered, normalized_radii);
    let half_size = max(center - vec2<f32>(radius), vec2<f32>(0.0));
    let q = abs(centered) - half_size;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn inner_corner_radii(corner_radii: vec4<f32>, border_widths: vec4<f32>) -> vec4<f32> {
    let top = border_widths.x;
    let right = border_widths.y;
    let bottom = border_widths.z;
    let left = border_widths.w;
    return max(
        corner_radii - vec4<f32>(max(left, top), max(right, top), max(right, bottom), max(left, bottom)),
        vec4<f32>(0.0),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = input.size.xy;
    let corner_radii = input.corner_radii;
    let border_alpha = input.border_color.a;
    let border_widths = max(input.border_widths, vec4<f32>(0.0)) * select(0.0, 1.0, border_alpha > 0.0);
    let distance = rounded_box_distance(input.local_position, size, corner_radii);
    let edge_alpha = clamp(0.5 - distance, 0.0, 1.0);
    let inner_size = max(size - vec2<f32>(border_widths.w + border_widths.y, border_widths.x + border_widths.z), vec2<f32>(0.0));
    let inner_position = input.local_position - vec2<f32>(border_widths.w, border_widths.x);
    let inner_distance = rounded_box_distance(inner_position, inner_size, inner_corner_radii(corner_radii, border_widths));
    let fill_alpha = clamp(0.5 - inner_distance, 0.0, 1.0);
    let fill_coverage = fill_alpha * input.fill_color.a;
    let border_coverage = max(edge_alpha - fill_alpha, 0.0) * border_alpha;
    let output_alpha = fill_coverage + border_coverage;
    let output_rgb = select(
        vec3<f32>(0.0),
        (input.fill_color.rgb * fill_coverage + input.border_color.rgb * border_coverage) / max(output_alpha, 0.000001),
        output_alpha > 0.0,
    );
    return vec4<f32>(output_rgb, output_alpha);
}
