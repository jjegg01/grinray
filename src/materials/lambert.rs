use cgmath::{ElementWise, InnerSpace, Vector3, Zero};
use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext, objects::{ObjectTransform, RTObject}, RTIntersection, Ray, TraceEvent, Tracer,
};

use super::Material;

/// Calculate next ray for the interaction between a Lambertian diffusor and a ray
pub(crate) fn lambert_next_ray(
    intersection: &RTIntersection,
    initial_depth: usize,
    rng: &mut Xoshiro256Plus,
) -> Option<Ray> {
    if initial_depth == 0 {
        return None;
    }
    let distr = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let unit_random = Vector3::new(distr.sample(rng), distr.sample(rng), distr.sample(rng)).normalize();
    Some(Ray {
        start: intersection.point,
        dir: (intersection.normal + unit_random).normalize(),
        depth: initial_depth,
    })
}

/// Material that implements a perfect Lambertian diffusor (i.e., a material that scatters incoming
/// rays according to Lambert's law)
#[derive(Clone)]
pub struct LambertMaterial {
    /// Color tint of the material (multiplicative)
    color: Vector3<f32>,
}

impl LambertMaterial {
    /// Create a new Lambertian material with a solid color (color is multiplied with ray color during
    /// raytracing)
    pub fn new(color: Vector3<f32>) -> Self {
        Self { color }
    }
}

impl<T: Tracer> Material<T> for LambertMaterial {
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
        let next_ray = <LambertMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            transform,
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
        _: &ObjectTransform,
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
        lambert_next_ray(intersection, ray.depth - 1, rng)
    }
}
