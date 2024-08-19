//! Material for rendering

mod checkerboard;
mod fresnel;
mod lambert;
mod lineargrin;

pub use checkerboard::*;
pub use fresnel::*;
pub use lambert::*;
pub use lineargrin::*;

use cgmath::Vector3;

use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext,
    objects::RTObject,
    ray::{RTIntersection, Ray},
    Tracer,
};

/// Trait for describing light matter interactions
pub trait Material<T: Tracer> {
    /// Interaction function for graphical raytracing, i.e., this function should return the color
    /// produced by the specified ray intersection
    fn interact(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32>;

    /// Interaction function for non-graphical raytracing, i.e., given the parameters of a ray
    /// entering an object, this function should compute the outgoing ray
    /// If more than one output ray is produced by the interaction (e.g., in the case of diffusive
    /// materials), this function should use the provided random number generator to sample the
    /// distribution of output rays
    fn next_ray(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray>;
}
