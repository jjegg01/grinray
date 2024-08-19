use cgmath::{ElementWise, InnerSpace, Vector3, Zero};
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext, objects::RTObject, util, RTIntersection, Ray, TraceEvent, Tracer,
};

use super::Material;

/// Checkerboard material that is mainly intended for the base plane so you can see refraction more nicely
#[derive(Clone)]
pub struct CheckerboardMaterial {
    color: Vector3<f32>,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

impl CheckerboardMaterial {
    /// Create a new material showing a checkerboard pattern.
    /// The pattern requires an origin, a direction and a color if you don't want white tiles
    pub fn new(color: Vector3<f32>, origin: Vector3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            color,
            origin,
            direction,
        }
    }
}

impl<T: Tracer> Material<T> for CheckerboardMaterial {
    fn interact(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        let next_ray = <CheckerboardMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            &mut ctx.rng,
            tracer,
            trace,
        );
        match next_ray {
            Some(next_ray) => self
                .color
                .mul_element_wise(next_ray.get_color(ctx, tracer, trace)),
            None => Vector3::zero(),
        }
    }

    fn next_ray(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        _: &(dyn RTObject + Send + Sync),
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Calculate checkerboard base color
        let point = util::vec3_64_to_32(&intersection.point);
        let normal = util::vec3_64_to_32(&intersection.normal);
        let plane_vec = self.origin - point;
        let direction2 = self.direction.cross(normal);
        let tex_x = plane_vec.dot(self.direction);
        let tex_y = plane_vec.dot(direction2);
        let sign_x = if tex_x < 0.0 { 1 } else { 0 };
        let sign_y = if tex_y < 0.0 { 1 } else { 0 };
        if ((tex_x as i32).abs() + (tex_y as i32).abs() + sign_x + sign_y) % 2 == 1 {
            // Diffuse reflection on "white" tiles
            tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
            super::lambert_next_ray(intersection, ray.depth - 1, rng)
        } else {
            // No exit ray on "black" tiles
            tracer.add_point(trace, TraceEvent::End, intersection.point);
            None
        }
    }
}
