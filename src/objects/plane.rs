use cgmath::{InnerSpace, Rotation, Vector3};

use crate::{RTIntersection, Ray, World};

use super::{ObjectTransform, RTObject};

/// An infinitely large plane
/// 
/// Without any transformation this object is identical to the XZ plane, i.e., it crosses the
/// coordinate origin and it is orthogonal to the Y-axis.
#[derive(Clone)]
pub struct Plane {}

impl Plane {
    pub fn new() -> Self {
        Self {}
    }
}

impl RTObject for Plane {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &Ray, world: &World) -> Option<RTIntersection> {
        // Discard rays that are too close to being parallel
        let origin = transform.translation;
        let normal = transform.rotation.rotate_vector(Vector3::unit_y());
        let nd = normal.dot(ray.dir);
        if nd.abs() == 0.0 {
            None
        } else {
            let ray_dist = (origin - ray.start).dot(normal) / nd;
            if ray_dist >= world.ray_distance_epsilon {
                let point = ray.start + ray_dist * ray.dir;
                // Set the intersection normal such that it always points "against" the direction
                // of the incoming ray
                let normal = if nd > 0. { -normal } else { normal };
                Some(RTIntersection {
                    ray_dist,
                    point,
                    normal,
                })
            } else {
                None
            }
        }
    }

    fn intersect_line(&self, transform: &ObjectTransform, ray: &Ray, _: &World) -> Option<RTIntersection> {
        // Discard rays that are too close to being parallel
        let origin = transform.translation;
        let normal = transform.rotation.rotate_vector(Vector3::unit_y());
        let nd = normal.dot(ray.dir);
        if nd.abs() == 0.0 {
            None
        } else {
            let ray_dist = (origin - ray.start).dot(normal) / nd;
            let point = ray.start + ray_dist * ray.dir;
            // If the intersection is at positive ray distances we use the same sign convention
            // for the normal as in the intersect_ray case. Otherwise, we flip the sign again.
            // This way, the reported normal of the intersection always points towards the start
            // point of the ray.
            let normal = if ray_dist > 0. {
                if nd > 0. {
                    -normal
                } else {
                    normal
                }
            } else {
                if nd > 0. {
                    normal
                } else {
                    -normal
                }
            };
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        }
    }
}
