mod cuboid;
mod plane;
mod sphere;

pub use cuboid::*;
pub use plane::*;
pub use sphere::*;

use cgmath::{InnerSpace, Quaternion, Vector3};

use crate::{RTIntersection, Ray};

/// Trait for objects that can be intersected by straight lines, i.e. objects that can be raytraced
pub trait RTObject {
    /// Closest intersection of a ray with this objects (but at least RAYDIST_EPSILON away from the
    /// start of the ray).
    ///
    /// This function does *not* consider intersections that are "behind" the start of the ray
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection>;
    /// Closest intersection with a line.
    ///
    /// This function also checks for intersections "behind" the start of the given ray and does
    /// *not* have a minimum distance between the ray start and the intersection point.
    fn intersect_line(&self, transform: &ObjectTransform, line: &Ray) -> Option<RTIntersection>;
    // /// Some materials may introduce material properties with a spatial dependency. This point is
    // /// used as a reference for these properties.
    // fn reference_point(&self) -> &Vector3<f64>;
}


/// Representation for a transformation of a geometric object consisting of a rotation followed by
/// a translation
#[derive(Clone)]
pub struct ObjectTransform {
    pub(crate) rotation: Quaternion<f64>,
    pub(crate) translation: Vector3<f64>
}

impl ObjectTransform {
    /// Takes a rotation (as a quarternion) and a translation (as a displacement vector) to
    /// construct a new object transformation
    /// 
    /// Note: if the quarternion is not normalized, this function will normalize it first
    pub fn new<R: Into<Quaternion<f64>>>(translation: Vector3<f64>, rotation: Option<R>) -> Self {
        let rotation: Quaternion<f64> = match rotation {
            Some(rotation) => rotation.into(),
            None => Quaternion::new(1.0, 0.0, 0.0, 0.0),
        };
        Self { rotation: rotation.normalize(), translation }
    }

    pub fn get_translation(&self) -> &Vector3<f64> {
        &self.translation
    }

    pub fn get_rotation(&self) -> &Quaternion<f64> {
        &self.rotation
    }
}