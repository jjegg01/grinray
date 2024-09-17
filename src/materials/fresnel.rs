use cgmath::{InnerSpace, Vector3, Zero};
use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext, objects::{ObjectTransform, RTObject}, RTIntersection, Ray, TraceEvent, Tracer,
};

use super::Material;

/// Perfectly smooth Fresnel refractor
#[derive(Clone)]
pub struct FresnelMaterial {
    /// Index inside the material
    index: f64,
    /// Index outside of the material
    outer_index: f64,
}

pub(crate) enum FresnelInteractionType {
    Reflection(Ray),
    Refraction(Ray),
}

// When you don't care about the interaction type, you can just into() the result into a normal ray
impl Into<Ray> for FresnelInteractionType {
    fn into(self) -> Ray {
        match self {
            FresnelInteractionType::Reflection(ray) => ray,
            FresnelInteractionType::Refraction(ray) => ray,
        }
    }
}

impl FresnelMaterial {
    /// Create a new material implementing the Fresnel equations for light-matter interaction
    ///
    /// You must specify the refractive index of the material itself *and* of the surrounding medium
    pub fn new(index: f64, outer_index: f64) -> Self {
        Self { index, outer_index }
    }

    pub(crate) fn fresnel_interaction(
        index: f64,
        outer_index: f64,
        incoming_ray_direction: Vector3<f64>,
        incoming_ray_depth: usize,
        intersection_point: Vector3<f64>,
        intersection_normal: Vector3<f64>,
        rng: &mut Xoshiro256Plus,
    ) -> FresnelInteractionType {
        // Get refractive indices on the incoming and outgoing side of the interface
        // as well as "canonical" normal vector that points towards the incoming ray
        let (n_in, n_out, normal) = if incoming_ray_direction.dot(intersection_normal) < 0. {
            // Ray entering object, i.e. surface normal points towards incoming ray
            (outer_index, index, intersection_normal)
        } else {
            // Ray leaving object
            (index, outer_index, -intersection_normal)
        };
        // -- Calculate reflected and transmitted ray intensities --
        let l_dot_n = incoming_ray_direction.dot(normal);
        let index_ratio = n_in / n_out;
        // Check for total reflection
        if index_ratio * index_ratio * (1. - l_dot_n * l_dot_n) > 1.0 {
            let dir = incoming_ray_direction - 2. * l_dot_n * normal;
            return FresnelInteractionType::Reflection(Ray {
                start: intersection_point,
                dir,
                depth: incoming_ray_depth - 1,
            });
        }
        // Dot product gives cosine of incident angle, but we need to flip the normal hence the sign
        let cosine_term = -l_dot_n; 
        // Since 1 - cos^2 = sin, this term contains the sine of the incident angle
        let sine_term = (1. - index_ratio * index_ratio * (1. - l_dot_n * l_dot_n)).sqrt(); 
        // Fresnel equations
        let reflectance_s = ((index_ratio * cosine_term - sine_term)
            / (index_ratio * cosine_term + sine_term))
            .powf(2.);
        let reflectance_p = ((index_ratio * sine_term - cosine_term)
            / (index_ratio * sine_term + cosine_term))
            .powf(2.);
        // TODO: track polarization
        let reflectance = (reflectance_p + reflectance_s) / 2.;
        // -- Monte-Carlo tracing --
        if rand::distributions::Uniform::new(0., 1.).sample(rng) < reflectance {
            // Reflection
            let dir = incoming_ray_direction - 2. * l_dot_n * normal;
            FresnelInteractionType::Reflection(Ray {
                start: intersection_point,
                dir,
                depth: incoming_ray_depth - 1,
            })
        } else {
            let dir = index_ratio * incoming_ray_direction - (index_ratio * l_dot_n + sine_term) * normal;
            // Refraction
            FresnelInteractionType::Refraction(Ray {
                start: intersection_point,
                dir,
                depth: incoming_ray_depth - 1,
            })
        }
    }
}

impl<T: Tracer> Material<T> for FresnelMaterial {
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
        // TODO: handle rays originating WITHIN the object
        if ray.dir.dot(intersection.normal) > 0. {
            unimplemented!("Ray may not start inside a Fresnel material (yet)")
        }
        <FresnelMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            transform,
            &mut ctx.rng,
            tracer,
            trace,
        )
        .map(|ray| ray.get_color(ctx, tracer, trace))
        .unwrap_or(Vector3::zero())
    }

    fn next_ray(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        transform: &ObjectTransform,
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Keep track of the current ray
        let mut ray = ray.clone();
        let mut intersection = intersection.clone();
        // Track if we are inside the object or not
        let mut inside = false;
        // Two possible ray paths:
        // 1) Reflection
        // 2) Refraction -> Reflection* -> Refraction
        loop {
            // Abort recursion if ray depth is exhausted
            if ray.depth == 0 {
                tracer.add_point(trace, TraceEvent::End, intersection.point);
                return None;
            }
            match Self::fresnel_interaction(self.index, self.outer_index, ray.dir, ray.depth, intersection.point, intersection.normal, rng)
            {
                FresnelInteractionType::Reflection(next_ray) => {
                    tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
                    // Inner reflections bounce on the inner surface
                    if inside {
                        ray = next_ray;
                        intersection = object.intersect_ray(transform, &ray).unwrap();
                    }
                    // Outer reflections don't enter the object at all
                    else {
                        break Some(next_ray);
                    }
                }
                FresnelInteractionType::Refraction(next_ray) => {
                    tracer.add_point(trace, TraceEvent::Refraction, intersection.point);
                    // First refraction enters the object
                    if !inside {
                        ray = next_ray;
                        intersection = object.intersect_ray(transform, &ray).unwrap();
                        inside = true;
                    }
                    // Second refractions exits the object
                    else {
                        break Some(next_ray);
                    }
                }
            }
        }
    }
}
