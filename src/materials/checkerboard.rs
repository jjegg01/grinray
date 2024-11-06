use cgmath::{ElementWise, InnerSpace, Rotation, Vector3, Zero};
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext, objects::{ObjectTransform, RTObject}, RTIntersection, Ray, TraceEvent, Tracer, World
};

use super::Material;

/// Checkerboard material that is mainly intended for the base plane so you can see refraction more nicely
#[derive(Clone)]
pub struct CheckerboardMaterial {
    color: Vector3<f32>,
    direction: Vector3<f64>,
}

impl CheckerboardMaterial {
    /// Create a new material showing a checkerboard pattern.
    /// The pattern requires a direction (basically the direction of one of the sides of the square
    /// tiles) and a color if you don't want white tiles
    /// 
    /// Note: The `direction` is understood to be in the **untransformed** coordinate system of
    /// each object, i.e., if you apply this to a `Plane` object, which in its untransformed 
    /// coordinate system is an XZ-plane, you should choose a `direction` in the XZ-plane as well,
    /// regardless of how you transform the plane later.
    pub fn new(color: Vector3<f32>, direction: Vector3<f64>) -> Self {
        Self {
            color,
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
        transform: &ObjectTransform,
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        let next_ray = <CheckerboardMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            transform,
            &ctx.world,
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
        transform: &ObjectTransform,
        _: &World,
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Rotate pattern direction with the rotation of the object
        let direction = transform.rotation.rotate_vector(self.direction);
        // Calculate checkerboard base color
        let plane_vec = transform.translation - intersection.point;
        let direction2 = direction.cross(intersection.normal);
        let tex_x = plane_vec.dot(direction);
        let tex_y = plane_vec.dot(direction2);
        let sign_x = if tex_x < 0.0 { 1 } else { 0 };
        let sign_y = if tex_y < 0.0 { 1 } else { 0 };
        if ((tex_x as i64).abs() + (tex_y as i64).abs() + sign_x + sign_y) % 2 == 1 {
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
