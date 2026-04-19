fn clip_alpha(clip_bounds: vec4<f32>, point: vec2<f32>) -> f32 {
    let min_bounds = clip_bounds.xy;
    let max_bounds = clip_bounds.xy + clip_bounds.zw;
    let inside = point.x >= min_bounds.x
        && point.x <= max_bounds.x
        && point.y >= min_bounds.y
        && point.y <= max_bounds.y;
    return select(0.0, 1.0, inside);
}
