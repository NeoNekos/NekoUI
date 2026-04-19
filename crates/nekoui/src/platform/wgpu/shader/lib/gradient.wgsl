fn sample_linear_gradient(
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    t: f32,
) -> vec4<f32> {
    return mix(start_color, end_color, clamp(t, 0.0, 1.0));
}
