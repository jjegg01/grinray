//! Utilities

use cgmath::Vector3;

pub(crate) fn color_vec3_to_u32(color: &Vector3<f32>) -> u32 {
    // NOTE: WE ASSUME LITTLE ENDIAN HERE
    let mut result = 255;
    result = (result << 8) + ((color.z).clamp(0., 0.999) * 256.) as u32;
    result = (result << 8) + ((color.y).clamp(0., 0.999) * 256.) as u32;
    (result << 8) + ((color.x).clamp(0., 0.999) * 256.) as u32
}
