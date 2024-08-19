use cgmath::{InnerSpace, Vector3};

use crate::{RTIntersection, Ray, RAYDIST_EPSILON};

use super::RTObject;

/// Infinite plane defined by an origin and a normal direction
#[derive(Clone)]
pub struct Plane {
    origin: Vector3<f64>,
    normal: Vector3<f64>,
}

impl Plane {
    pub fn new(origin: Vector3<f64>, normal: Vector3<f64>) -> Self {
        Self { origin, normal }
    }
}

impl RTObject for Plane {
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection> {
        // Discard rays that are too close to being parallel
        let nd = self.normal.dot(ray.dir);
        if nd.abs() == 0.0 {
            None
        } else {
            let ray_dist = (self.origin - ray.start).dot(self.normal) / nd;
            if ray_dist >= RAYDIST_EPSILON {
                let point = ray.start + ray_dist * ray.dir;
                let normal = self.normal;
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

    fn intersect_line(&self, _ray: &Ray) -> Option<RTIntersection> {
        todo!()
    }

    fn reference_point(&self) -> &Vector3<f64> {
        &self.origin
    }
}
