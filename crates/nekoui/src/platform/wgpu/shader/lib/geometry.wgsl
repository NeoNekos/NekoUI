const UNIT_CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
);

fn rect_to_ndc(point: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (point.x / view.viewport.x) * 2.0 - 1.0,
        1.0 - (point.y / view.viewport.y) * 2.0,
    );
}
