//! Implementation of an axis-aligned box object

use cgmath::Vector3;

use super::{ObjectTransform, RTObject};

pub struct AxisAlignedBox {
    length_x: f64,
    length_y: f64,
    length_z: f64
}

impl AxisAlignedBox {
    pub fn new(length_x: f64, length_y: f64, length_z: f64) -> Self {
        Self { length_x, length_y, length_z }
    }
}

impl RTObject for AxisAlignedBox {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &crate::Ray) -> Option<crate::RTIntersection> {
        todo!()
    }

    fn intersect_line(&self, transform: &ObjectTransform, line: &crate::Ray) -> Option<crate::RTIntersection> {
        todo!()
    }

    // fn reference_point(&self) -> &cgmath::Vector3<f64> {
    //     todo!()
    //     // Vector3::new(
    //     //     (self.extent_x.0 + self.extent_x.1) / 2.,
    //     //      y, z)
    // }
}