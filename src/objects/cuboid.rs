//! Implementation of rectangular cuboids

use core::f64;

use cgmath::{Rotation, Vector3};

use crate::{util::{max_pair, min_pair, minmax_pair}, RTIntersection, Ray, RAYDIST_EPSILON};

use super::{ObjectTransform, RTObject};

/// Rectangular cuboid with given dimensions
/// 
/// Without any transformation this object is centered about the origin and its faces are
/// pependicular to the coordinate axes.
#[derive(Clone)]
pub struct Cuboid {
    pub length_x: f64,
    pub length_y: f64,
    pub length_z: f64
}


#[derive(Clone, Copy, Debug)]
enum CubePlane { XLo, XHi, YLo, YHi, ZLo, ZHi }

impl Cuboid {
    pub fn new(length_x: f64, length_y: f64, length_z: f64) -> Self {
        Self { length_x, length_y, length_z }
    }

    /// Calculate the part of a ray that is inside this cuboid in terms of distances from the start
    /// of the ray. Returns a tuple with the lower and upper bound of the distances as well as the
    /// cube planes that are intersected at these limiting distances. If the first distance is
    /// larger than the second distance, there is no intersection.
    fn calc_raydist_bounds(&self, transform: &ObjectTransform, ray: &crate::Ray) -> ((f64, CubePlane), (f64, CubePlane)) {
        // This is basically the algorithm described here: https://tavianator.com/2011/ray_box.html

        // Transform ray to object frame
        let ray_object_frame = Ray {
            start: transform.rotation.conjugate().rotate_vector(ray.start - transform.translation),
            dir: transform.rotation.conjugate().rotate_vector(ray.dir),
            depth: ray.depth,
        };

        // Calculate ray distances to intersection points with each plane
        // Note
        let upper_bounds = Vector3::new(self.length_x / 2., self.length_y / 2., self.length_z / 2.);
        let lower_bounds = -upper_bounds;
        let s_upper = Vector3::new(
            (upper_bounds.x - ray_object_frame.start.x) / ray_object_frame.dir.x,
            (upper_bounds.y - ray_object_frame.start.y) / ray_object_frame.dir.y,
            (upper_bounds.z - ray_object_frame.start.z) / ray_object_frame.dir.z,
        );
        let s_lower = Vector3::new(
            (lower_bounds.x - ray_object_frame.start.x) / ray_object_frame.dir.x,
            (lower_bounds.y - ray_object_frame.start.y) / ray_object_frame.dir.y,
            (lower_bounds.z - ray_object_frame.start.z) / ray_object_frame.dir.z,
        );
        // If the ray intersects the box, there is a ray distance s_min where it enters the box and
        // a distance s_max where it exits the box. We can use the 6 ray distances we calculated
        // above as bounds:
        // min(s_upper.x, s_lower.x) <= s_min <= s_max <= max(s_upper.x, s_lower.x)
        // (same for y and z)
        // To keep track of the cube plane we intersected (for surface normal) we bundle the ray
        // distances and the corresponding cube planes into a tuple
        let (s_min_x, s_max_x) = minmax_pair(&(s_lower.x, CubePlane::XLo), &(s_upper.x, CubePlane::XHi));
        let (s_min_y, s_max_y) = minmax_pair(&(s_lower.y, CubePlane::YLo), &(s_upper.y, CubePlane::YHi));
        let (s_min_z, s_max_z) = minmax_pair(&(s_lower.z, CubePlane::ZLo), &(s_upper.z, CubePlane::ZHi));
        let s_min = max_pair(&s_min_x, &max_pair(&s_min_y, &s_min_z));
        let s_max = min_pair(&s_max_x, &min_pair(&s_max_y, &s_max_z));
        
        (s_min, s_max)
    }

    /// Construct a proper RTIntersection from a tuple containing the ray distance at which the
    /// cuboid is entered/exited and the cube plane that is intersected
    fn raydist_to_intersection(&self, s_intersect: (f64, CubePlane), transform: &ObjectTransform, ray: &crate::Ray) -> RTIntersection {
        let ray_dist = s_intersect.0;
        // To calculate the intersection point we can directly use the lab frame since distances
        // are invariant under translations and rotations
        let point = ray.start + ray_dist * ray.dir;
        // We need to rotate the normal from the object frame to the lab frame though
        let normal_object_frame = match s_intersect.1 {
            CubePlane::XLo => -Vector3::unit_x(),
            CubePlane::XHi => Vector3::unit_x(),
            CubePlane::YLo => -Vector3::unit_y(),
            CubePlane::YHi => Vector3::unit_y(),
            CubePlane::ZLo => -Vector3::unit_z(),
            CubePlane::ZHi => Vector3::unit_z(),
        };
        let normal = transform.rotation.rotate_vector(normal_object_frame);

        RTIntersection {
            ray_dist,
            point,
            normal,
        }
    }
}

impl RTObject for Cuboid {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &crate::Ray) -> Option<crate::RTIntersection> {
        // Calculate the upper and lower bounds on the ray distance
        let (s_min, s_max) = self.calc_raydist_bounds(transform, ray);

        // If the interval [s_min,s_max] is empty, there is no intersection
        // If the interval is not empty, but s_max < 0, the intersection is behind the ray
        // In both cases we do not have a proper intersection to return
        if s_max.0 < s_min.0 || s_max.0 < 0. {
            None
        }
        else {
            // If s_min is negative, we are inside the cube, i.e. the intersection is at s_max.
            // Otherwise, it is at s_min
            let s_intersect = if s_min.0 < RAYDIST_EPSILON { s_max } else { s_min };

            // Ignore intersection if it is too close
            if s_intersect.0 < RAYDIST_EPSILON {
                return None
            }

            Some(self.raydist_to_intersection(s_intersect, transform, ray))
        }
    }

    fn intersect_line(&self, transform: &ObjectTransform, line: &crate::Ray) -> Option<crate::RTIntersection> {
        // Calculate the upper and lower bounds on the ray distance
        let (s_min, s_max) = self.calc_raydist_bounds(transform, line);

        // If the interval [s_min,s_max] is empty, there is no intersection
        if s_max.0 < s_min.0 {
            None
        }
        else {
            // Choose the intersection point that is closest to the start of the ray
            let s_intersect = if s_min.0.abs() < s_max.0.abs() { s_min } else { s_max };

            Some(self.raydist_to_intersection(s_intersect, transform, line))
        }
    }
}