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
        let center = transform.translation;
        let a = ray.start - center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        if discriminant >= 0.0 {
            let discriminant = discriminant.sqrt();
            let solution1 = -ad - discriminant;
            let solution2 = -ad + discriminant;
            let ray_dist = if solution1 >= RAYDIST_EPSILON {
                solution1
            } else if solution2 >= RAYDIST_EPSILON {
                solution2
            } else {
                return None;
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

    fn intersect_line(&self, transform: &ObjectTransform, ray: &Ray) -> Option<RTIntersection> {
        let center = transform.translation;
        let a = ray.start - center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        if discriminant >= 0.0 {
            let discriminant = discriminant.sqrt();
            let solution1 = -ad - discriminant;
            let solution2 = -ad + discriminant;
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

    // fn reference_point(&self) -> &Vector3<f64> {
    //     &self.center
    // }
}
