//! Definition of a sphere via a signed distance function

use cgmath::{MetricSpace, Vector3, Zero};

use super::{AABox, RescaleAbsoluteUniform, SDFObject, SURFACE_EPSILON};

/// A sphere with a given radius
pub struct Sphere {
    pub radius: f64,
}

impl Sphere {
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

impl SDFObject for Sphere {
    fn sdf(&self, position: &cgmath::Vector3<f64>) -> f64 {
        position.distance(Vector3::zero()) - self.radius
    }

    fn bounding_box(&self) -> super::AABox {
        AABox {
            xlo: -self.radius - SURFACE_EPSILON,
            xhi: self.radius + SURFACE_EPSILON,
            ylo: -self.radius - SURFACE_EPSILON,
            yhi: self.radius + SURFACE_EPSILON,
            zlo: -self.radius - SURFACE_EPSILON,
            zhi: self.radius + SURFACE_EPSILON,
        }
    }
}

impl RescaleAbsoluteUniform for Sphere {
    fn rescale_absolute(&mut self, size_change: f64) {
        self.radius += size_change / 2.;
    }
}