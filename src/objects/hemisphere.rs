use core::f64;

use cgmath::{InnerSpace, Rotation, Vector3, Zero};

use crate::{RTIntersection, Ray, RAYDIST_EPSILON};

use super::{ObjectTransform, RTObject};

/// Hemisphere with a given radius. The cut-plane is the XZ plane with the
/// hemisphere being in the space of positive Y coordinates.
///
/// Without any transformation this object is centered about the origin.
#[derive(Clone)]
pub struct Hemisphere {
    pub radius: f64,
}

impl Hemisphere {
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

struct HemisphereIntersection {
    ray_dist: f64,
    normal_object_frame: Vector3<f64>,
    valid: bool,
}

impl Hemisphere {
    fn get_intersection_candidates(
        &self,
        transform: &ObjectTransform,
        ray: &Ray,
    ) -> Option<[HemisphereIntersection; 3]> {
        // Transform to object coordinates
        let ray_object_frame = transform.ray_to_object_frame(&ray);
        // Test first if the ray intersects the sphere
        let a = ray_object_frame.start;
        let ad = a.dot(ray_object_frame.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        // Check if we intersect the sphere
        if discriminant >= 0.0 {
            // The first two candidates are the intersection with the full sphere
            let ray_dist1 = -ad - discriminant.sqrt();
            let ray_dist2 = -ad + discriminant.sqrt();
            let intersection_objframe_1 = ray_object_frame.start + ray_dist1 * ray_object_frame.dir;
            let intersection_objframe_2 = ray_object_frame.start + ray_dist2 * ray_object_frame.dir;
            // The third candidate is the intersection with the plane
            let ray_dist3 = -ray_object_frame.start.y / ray_object_frame.dir.y;
            let intersection_objframe_3 = ray_object_frame.start + ray_dist3 * ray_object_frame.dir;
            // Return all candidates, but flag the invalid ones
            Some([
                HemisphereIntersection {
                    ray_dist: ray_dist1,
                    normal_object_frame: intersection_objframe_1,
                    valid: intersection_objframe_1.y >= 0.,
                },
                HemisphereIntersection {
                    ray_dist: ray_dist2,
                    normal_object_frame: intersection_objframe_2,
                    valid: intersection_objframe_2.y >= 0.,
                },
                HemisphereIntersection {
                    ray_dist: ray_dist3,
                    normal_object_frame: -Vector3::unit_y(),
                    valid: (intersection_objframe_3.x * intersection_objframe_3.x
                        + intersection_objframe_3.z * intersection_objframe_3.z)
                        <= self.radius * self.radius,
                },
            ])

        // If there is no intersection with the full sphere, there is also no
        // intersection with the hemisphere
        } else {
            None
        }
    }
}

impl RTObject for Hemisphere {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Get candidates for intersection and select the closest one in the forward direction
        let candidates = self.get_intersection_candidates(transform, ray)?;
        let mut ray_dist = f64::INFINITY;
        let mut normal_object_frame = Vector3::zero();
        for candidate in candidates {
            if candidate.valid && candidate.ray_dist >= RAYDIST_EPSILON && candidate.ray_dist < ray_dist {
                ray_dist = candidate.ray_dist;
                normal_object_frame = candidate.normal_object_frame;
            }
        }
        // No valid candidate found (e.g., ray points away from (hemi-)sphere)
        if ray_dist.is_infinite() {
            return None;
        };
        // Return final intersection
        let point = ray.start + ray_dist * ray.dir;
        let normal = transform.rotation.rotate_vector(normal_object_frame);
        Some(RTIntersection {
            ray_dist,
            point,
            normal,
        })
    }

    fn intersect_line(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Get candidates for intersection and select the closest one
        let candidates = self.get_intersection_candidates(transform, ray)?;
        let mut ray_dist = f64::INFINITY;
        let mut normal_object_frame = Vector3::zero();
        for candidate in candidates {
            if candidate.valid && candidate.ray_dist.abs() < ray_dist.abs() {
                ray_dist = candidate.ray_dist;
                normal_object_frame = candidate.normal_object_frame;
            }
        }
        // No valid candidate found (is this even possible?)
        if ray_dist.is_infinite() {
            return None;
        };
        // Return final intersection
        let point = ray.start + ray_dist * ray.dir;
        let normal = transform.rotation.rotate_vector(normal_object_frame);
        Some(RTIntersection {
            ray_dist,
            point,
            normal,
        })
    }
}
