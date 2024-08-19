use cgmath::Vector3;

/// Lower precision from Vector3<f64> to Vector3<f32>
pub(crate) fn vec3_64_to_32(v: &Vector3<f64>) -> Vector3<f32> {
    Vector3 {
        x: v.x as f32,
        y: v.y as f32,
        z: v.z as f32,
    }
}
