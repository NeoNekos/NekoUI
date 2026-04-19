fn rounded_rect_sdf(local_pos: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(local_pos - half_size) - (half_size - vec2<f32>(radius));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}
