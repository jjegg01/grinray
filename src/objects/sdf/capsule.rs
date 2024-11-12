use core::f64;

use cgmath::Vector3;

use super::{AABox, RescaleAbsoluteUniform, SDFObject, SURFACE_EPSILON};

/// A capsule (also called a spherocylinder) with a given radius and length. The
/// length in measured from center to center of the hemisphere caps, i.e., the
/// actual bounding box is larger by twice the radius.
///
/// Without any transformation this object is centered about the origin and the
/// capsule extends in the Y-direction.
#[derive(Clone)]
pub struct Capsule {
    pub radius: f64,
    pub length: f64,
}

impl Capsule {
    pub fn new(radius: f64, length: f64) -> Self {
        Self { radius, length }
    }
}

impl SDFObject for Capsule {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        // We can reduce the problem to 2D:
        // One coordinate is the distance from the axis d_xz, the other is the y-coordinate
        // Then, we observe that for |y| < h, the SDF ist just d_xz, while for |y| > h, the SDF is
        // sqrt(d_xz^2 + (|y|-h)^2). By clamping (|y|-h) to a lower bound of zero,
        // we can unify both cases:
        (position.x * position.x
            + position.z * position.z
            + (position.y.abs() - self.length / 2.)
                .clamp(0., f64::INFINITY)
                .powi(2))
        .sqrt()
            - self.radius
    }

    fn bounding_box(&self) -> super::AABox {
        AABox {
            xlo: -self.radius - SURFACE_EPSILON,
            xhi: self.radius + SURFACE_EPSILON,
            ylo: -(self.length + self.radius) - SURFACE_EPSILON,
            yhi: self.length + self.radius + SURFACE_EPSILON,
            zlo: -self.radius - SURFACE_EPSILON,
            zhi: self.radius + SURFACE_EPSILON,
        }
    }
}

impl RescaleAbsoluteUniform for Capsule {
    fn rescale_absolute(&mut self, size_change: f64) {
        self.radius += size_change / 2.;
    }
}
