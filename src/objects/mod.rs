mod plane;
mod sphere;

pub use plane::*;
pub use sphere::*;

use cgmath::Vector3;

use crate::{RTIntersection, Ray};

/// Trait for objects that can be intersected by straight lines, i.e. objects that can be raytraced
pub trait RTObject {
    /// Closest intersection of a ray with this objects (but at least RAYDIST_EPSILON away from the
    /// start of the ray).
    ///
    /// This function does *not* consider intersections that are "behind" the start of the ray
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection>;
    /// Closest intersection with a line.
    ///
    /// This function also checks for intersections "behind" the start of the given ray and does
    /// *not* have a minimum distance between the ray start and the intersection point.
    fn intersect_line(&self, line: &Ray) -> Option<RTIntersection>;
    /// Some materials may introduce material properties with a spatial dependency. This point is
    /// used as a reference for these properties.
    fn reference_point(&self) -> &Vector3<f64>;
}
