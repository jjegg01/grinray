use cgmath::{InnerSpace, Vector3};

use crate::{RTIntersection, Ray, RAYDIST_EPSILON};

use super::RTObject;

/// Sphere centered around a point with a given radius
#[derive(Clone)]
pub struct Sphere {
    pub center: Vector3<f64>,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl RTObject for Sphere {
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection> {
        let a = ray.start - self.center;
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
            let normal = (point - self.center).normalize();
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        } else {
            None
        }
    }

    fn intersect_line(&self, ray: &Ray) -> Option<RTIntersection> {
        let a = ray.start - self.center;
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
            let normal = (point - self.center).normalize();
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        } else {
            None
        }
    }

    fn reference_point(&self) -> &Vector3<f64> {
        &self.center
    }
}
