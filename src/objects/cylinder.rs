use core::f64;

use cgmath::{InnerSpace, Rotation, Vector2, Vector3};

use crate::{RTIntersection, Ray, RAYDIST_EPSILON};

use super::{ObjectTransform, RTObject};

/// A capsule (also called a spherocylinder) with a given radius and length. The
/// length in measured from center to center of the hemisphere caps, i.e., the
/// actual bounding box is larger by twice the radius.
///
/// Without any transformation this object is centered about the origin and the
/// capsule extends in the Y-direction.
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

enum CylinderIntersectionKind { Mantle, Top, Bottom }
struct CylinderIntersection {
    ray_dist: f64,
    intersection_object_frame: Vector3<f64>,
    valid: bool,
    kind: CylinderIntersectionKind
}
impl CylinderIntersection {
    fn mantle(ray_dist: f64, intersection_object_frame: Vector3<f64>) -> Self {
        Self { ray_dist, intersection_object_frame, valid: true, kind: CylinderIntersectionKind::Mantle}
    }
    fn top(ray_dist: f64, intersection_object_frame: Vector3<f64>) -> Self {
        Self { ray_dist, intersection_object_frame, valid: true, kind: CylinderIntersectionKind::Top}
    }
    fn bottom(ray_dist: f64, intersection_object_frame: Vector3<f64>) -> Self {
        Self { ray_dist, intersection_object_frame, valid: true, kind: CylinderIntersectionKind::Bottom}
    }
}

impl Cylinder {
    /// Helper function for calculating ray distance to intersection with cylinder bases
    fn intersect_base_raydist(base_y: f64, ray_object_frame: &Ray) -> f64 {
        (base_y - ray_object_frame.start.y) / ray_object_frame.dir.y
    }

    /// Calculate the intersection of a ray (object frame!) and the infinite extension of this cylinder
    /// Returns the distances of the mantle intersection points from the start of the ray, if these
    /// points exist
    fn get_ray_distances_infinite_mantle(&self, ray_object_frame: &Ray) -> Option<(f64, f64)> {
        // First project the ray and the cylinder down to the XZ plane
        let ray_dir_2d = Vector2::new(ray_object_frame.dir.x, ray_object_frame.dir.z);
        let ray_start_2d = Vector2::new(ray_object_frame.start.x, ray_object_frame.start.z);
        // Next we solve for the ray distance (i.e., solve a quadratic equation)
        let start_dir_dot_dist_2d = ray_dir_2d.dot(ray_start_2d) / ray_dir_2d.dot(ray_dir_2d);
        let discriminant = start_dir_dot_dist_2d * start_dir_dot_dist_2d
            + (self.radius * self.radius - ray_start_2d.dot(ray_start_2d))
                / ray_dir_2d.dot(ray_dir_2d);
        // No solution if the discriminant is negative
        if discriminant < 0. {
            return None;
        }
        let ray_dist_1 = -start_dir_dot_dist_2d - discriminant.sqrt();
        let ray_dist_2 = -start_dir_dot_dist_2d + discriminant.sqrt();
        Some((ray_dist_1, ray_dist_2))
    }

    fn get_intersection_candidates(&self, transform: &ObjectTransform, ray: &Ray) -> Option<[CylinderIntersection;4]> {
        // Rotate ray to object frame
        let ray_object_frame = transform.ray_to_object_frame(&ray);
        // Start with intersections with an infinitely extended cylinder
        let (ray_dist_1, ray_dist_2) = self.get_ray_distances_infinite_mantle(&ray_object_frame)?;
        // Intersection distances with the top and bottom base
        let ray_dist_top_base = Self::intersect_base_raydist(self.length / 2., &ray_object_frame);
        let ray_dist_btm_base = Self::intersect_base_raydist(-self.length / 2., &ray_object_frame);
        // Calculate intersection points in object coordinates
        let intersection_object_frame_1 =
            ray_object_frame.start + ray_dist_1 * ray_object_frame.dir;
        let intersection_object_frame_2 =
            ray_object_frame.start + ray_dist_2 * ray_object_frame.dir;
        let intersection_object_frame_top_base =
            ray_object_frame.start + ray_dist_top_base * ray_object_frame.dir;
        let intersection_object_frame_btm_base =
            ray_object_frame.start + ray_dist_btm_base * ray_object_frame.dir;
        // Gather candidates for the true intersection (bool at the end is used to invalidate
        // intersections outside the finite cylinder)
        let mut candidates = [
            CylinderIntersection::mantle(ray_dist_1, intersection_object_frame_1),
            CylinderIntersection::mantle(ray_dist_2, intersection_object_frame_2),
            CylinderIntersection::bottom(ray_dist_btm_base, intersection_object_frame_btm_base),
            CylinderIntersection::top(ray_dist_top_base, intersection_object_frame_top_base),
        ];
        // Validate all intersections
        for candidate in &mut candidates {
            match candidate.kind {
                CylinderIntersectionKind::Mantle => {
                    // Mantle intersections are only valid if the Y-coordinate is within [-L/2,+L/2]
                    // with L being the length of the cylinder
                    candidate.valid = candidate.intersection_object_frame.y.abs() <= self.length / 2.;
                },
                CylinderIntersectionKind::Top | CylinderIntersectionKind::Bottom => {
                    // Base intersections are only valid if the distance to the Y-axis is less than
                    // the radius of the cylinder
                    let x = candidate.intersection_object_frame.x;
                    let z = candidate.intersection_object_frame.z;
                    let r_sqr = x*x + z*z;
                    candidate.valid = r_sqr <= self.radius * self.radius;
                },
            };
        }
        Some(candidates)
    }

    fn cylinder_intersection_to_rt_intersection(intersection_final: CylinderIntersection, transform: &ObjectTransform, ray: &Ray) -> RTIntersection {
        let ray_dist = intersection_final.ray_dist;
        let intersection_point = ray.start + intersection_final.ray_dist * ray.dir;
        let normal_object_frame = match intersection_final.kind {
            CylinderIntersectionKind::Top => Vector3::new(0., 1., 0.),
            CylinderIntersectionKind::Bottom => Vector3::new(0., -1., 0.),
            CylinderIntersectionKind::Mantle => {
                Vector3::new(
                    intersection_final.intersection_object_frame.x, 
                    0.,
                    intersection_final.intersection_object_frame.z
                ).normalize()
            },
        };
        // Transform normal to world coordinates
        let normal = transform.get_rotation().rotate_vector(normal_object_frame);
        RTIntersection {
            ray_dist, point: intersection_point, normal
        }
    }
}

impl RTObject for Cylinder {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Get intersection candidates
        let candidates = self.get_intersection_candidates(transform, ray)?;
        // Select the closest valid intersection that has a positive ray distance
        let mut intersection_final = None;
        let mut max_dist = f64::INFINITY;
        for candidate in candidates {
            if candidate.valid && candidate.ray_dist > RAYDIST_EPSILON && candidate.ray_dist.abs() < max_dist {
                max_dist = candidate.ray_dist.abs();
                intersection_final = Some(candidate);
            }
        }
        let intersection_final = intersection_final?;
        // Process the "winning" intersection
        Some(Self::cylinder_intersection_to_rt_intersection(intersection_final, transform, ray))
    }

    fn intersect_line(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Get intersection candidates
        let candidates = self.get_intersection_candidates(transform, ray)?;
        // Select the closest valid intersection
        let mut intersection_final = None;
        let mut max_dist = f64::INFINITY;
        for candidate in candidates {
            if candidate.valid && candidate.ray_dist.abs() < max_dist {
                max_dist = candidate.ray_dist.abs();
                intersection_final = Some(candidate);
            }
        }
        let intersection_final = intersection_final?;
        // Process the "winning" intersection
        Some(Self::cylinder_intersection_to_rt_intersection(intersection_final, transform, ray))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cgmath::MetricSpace;
    const EPSILON: f64 = 1e-7;

    #[test]
    fn test_cylinder_rays_inside() {
        let cylinder = Cylinder::new(2.0, 2.0);
        let transform = ObjectTransform::identity();
        let ray_mantle = Ray {
            start: (0., 0., 0.).into(),
            dir: Vector3::new(1., 0.25, 1.).normalize(),
            depth: 42,
        };
        let intersection_mantle = cylinder.intersect_ray(&transform, &ray_mantle).unwrap();
        assert!((intersection_mantle.point.x - 2f64.sqrt()).abs() < EPSILON);
        assert!((intersection_mantle.point.z - 2f64.sqrt()).abs() < EPSILON);
        assert!(
            (ray_mantle.dir * intersection_mantle.ray_dist).distance(intersection_mantle.point)
                < EPSILON
        );
        let ray_top = Ray {
            start: (0., 0., 0.).into(),
            dir: Vector3::new(-1., 2., 1.).normalize(),
            depth: 42,
        };
        let intersection_top = cylinder.intersect_ray(&transform, &ray_top).unwrap();
        assert!((intersection_top.point.y - 1.0).abs() < EPSILON);
        assert!(
            (ray_mantle.dir * intersection_mantle.ray_dist).distance(intersection_mantle.point)
                < EPSILON
        );
        let ray_bottom = Ray {
            start: (0., 0., 0.).into(),
            dir: Vector3::new(1., -2., -1.).normalize(),
            depth: 42,
        };
        let intersection_top = cylinder.intersect_ray(&transform, &ray_bottom).unwrap();
        assert!((intersection_top.point.y + 1.0).abs() < EPSILON);
        assert!(
            (ray_mantle.dir * intersection_mantle.ray_dist).distance(intersection_mantle.point)
                < EPSILON
        );
    }
}
