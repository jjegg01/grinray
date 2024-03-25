use std::fmt::Debug;
use cgmath::{Vector3, Zero};
use rand_xoshiro::Xoshiro256Plus;

use crate::scene::Scene;

/// Representation of a single light ray with a depth counter
#[derive(Clone)]
pub struct Ray {
    /// Start point of the ray
    pub start: Vector3<f64>,
    /// Ray direction
    pub dir: Vector3<f64>,
    /// Ray depth counter
    pub depth: usize
}

/// Representation of an intersection between a light ray and an object
#[derive(Debug,Clone)]
pub struct RTIntersection {
    /// Distance from the ray origin to the intersection point
    pub ray_dist: f64,
    /// Point where the ray and the object intersect
    pub point: Vector3<f64>,
    /// Surface normal of the object at the intersection point
    pub normal: Vector3<f64>
}

impl Debug for Ray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ray(start: ({}, {}, {}), dir: ({}, {}, {}))", self.start.x, self.start.y, self.start.z, self.dir.x, self.dir.y, self.dir.z)
    }
}

/// Context object for probabilistic raytracing in a geometric scene of objects
pub struct RayGraphicsContext<'a> {
    pub(crate) scene: &'a Scene,
    pub(crate) rng: Xoshiro256Plus
}

impl Ray {
    pub(crate) fn get_color(&self, ctx: &mut RayGraphicsContext) -> Vector3<f32> {
        // Discard rays that have exhausted the depth limit
        if self.depth == 0 {
            Vector3::zero()
        }
        else {
            match ctx.scene.cast_ray(self) {
                Some((obj_id, intersection)) => {
                    let material = ctx.scene.get_object_material(obj_id);
                    let obj = ctx.scene.get_object(obj_id.0).unwrap();
                    material.interact(self, &intersection, obj, ctx)
                },
                None => ctx.scene.get_sky_color().clone(),
            }
        }
    }
}
