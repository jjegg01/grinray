use core::f64;

use cgmath::{MetricSpace, Vector2, Vector3, Zero};

use super::{AABox, SDFObject, SURFACE_EPSILON};

/// A cylinder with a given radius and length.
///
/// Without any transformation this object is centered about the origin and the
/// cylinder extends in the Y-direction.
#[derive(Clone)]
pub struct Cylinder {
    pub radius: f64,
    pub length: f64,
}

impl Cylinder {
    pub fn new(radius: f64, length: f64) -> Self {
        Self { radius, length }
    }
}

impl SDFObject for Cylinder {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        // We can reduce the problem to 2D:
        // One coordinate is the distance from the axis d_xz, the other is the y-coordinate
        // Then, the problem reduces to the SDF of a rectangle (see cuboid for the 3D case)
        let d_xz = (position.x * position.x + position.z * position.z).sqrt();
        let d = Vector2::new(d_xz - self.radius, position.y.abs() - self.length / 2.);
        Vector2::new(d.x.max(0.), d.y.max(0.)).distance(Vector2::zero()) + d.x.max(d.y).min(0.)
    }

    fn bounding_box(&self) -> super::AABox {
        AABox {
            xlo: -self.radius - SURFACE_EPSILON,
            xhi: self.radius + SURFACE_EPSILON,
            ylo: -self.length - SURFACE_EPSILON,
            yhi: self.length + SURFACE_EPSILON,
            zlo: -self.radius - SURFACE_EPSILON,
            zhi: self.radius + SURFACE_EPSILON,
        }
    }
}
