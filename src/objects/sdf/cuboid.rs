//! Definition of a cuboid via a signed distance function

use cgmath::{MetricSpace, Vector3, Zero};

use super::{AABox, SDFObject, SURFACE_EPSILON};

/// Rectangular cuboid with given dimensions
///
/// Without any transformation this object is centered about the origin and its faces are
/// pependicular to the coordinate axes.
pub struct Cuboid {
    pub length_x: f64,
    pub length_y: f64,
    pub length_z: f64,
}

impl Cuboid {
    pub fn new(length_x: f64, length_y: f64, length_z: f64) -> Self {
        Self {
            length_x,
            length_y,
            length_z,
        }
    }
}

impl SDFObject for Cuboid {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        // Vector of distances to the cuboid planes (mapped to 1st quadrant via abs)
        let d = Vector3::new(
            position.x.abs() - self.length_x / 2.,
            position.y.abs() - self.length_y / 2.,
            position.z.abs() - self.length_z / 2.,
        );
        // Contribution for points outside the cube (clamped to zero otherwise)
        Vector3::new(d.x.max(0.), d.y.max(0.), d.z.max(0.))
            .distance(Vector3::zero())
        // Contribution for points inside the cube (clamped to zero otherwise)
        + d.x.max(d.y.max(d.z)).min(0.)
    }

    fn bounding_box(&self) -> super::AABox {
        AABox {
            xlo: -self.length_x - SURFACE_EPSILON,
            xhi: self.length_x + SURFACE_EPSILON,
            ylo: -self.length_y - SURFACE_EPSILON,
            yhi: self.length_y + SURFACE_EPSILON,
            zlo: -self.length_z - SURFACE_EPSILON,
            zhi: self.length_z + SURFACE_EPSILON,
        }
    }
}
