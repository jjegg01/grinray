use cgmath::InnerSpace;

use crate::{RTIntersection, Ray, RAYDIST_EPSILON};

use super::{ObjectTransform, RTObject};

/// Sphere with a given radius
/// 
/// Without any transformation this object is centered about the origin.
#[derive(Clone)]
pub struct Sphere {
    pub radius: f64,
}

impl Sphere {
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

impl RTObject for Sphere {
    fn intersect_ray(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Inserting the parametric description of a ray (start at r_0, direction d)
        // r(s) = r_0 + s * d
        // into the definition of a sphere centered at r_c with radius R
        // || r - r_c || = R
        // leads to a quadratic equation that we solve here
        let center = transform.translation;
        let a = ray.start - center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        // Does a real solution exist?
        if discriminant >= 0.0 {
            let solution1 = -ad - discriminant.sqrt();
            let solution2 = -ad + discriminant.sqrt();
            // Ignore solutions that are too close to the start of the ray
            let ray_dist = if solution1 >= RAYDIST_EPSILON {
                solution1
            } else if solution2 >= RAYDIST_EPSILON {
                solution2
            } else {
                return None;
            };
            let point = ray.start + ray_dist * ray.dir;
            // The surface normal for a sphere is parallel to the distance vector from center
            let normal = (point - center).normalize();
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        } else {
            None
        }
    }

    fn intersect_line(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        // Basically the same as intersect_ray but with a different logic on how
        // we select the solution to return
        let center = transform.translation;
        let a = ray.start - center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        if discriminant >= 0.0 {
            let discriminant = discriminant.sqrt();
            let solution1 = -ad - discriminant;
            let solution2 = -ad + discriminant;
            // Choose the closest intersection
            let ray_dist = if solution1.abs() < solution2.abs() {
                solution1
            } else {
                solution2
            };
            let point = ray.start + ray_dist * ray.dir;
            let normal = (point - center).normalize();
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
