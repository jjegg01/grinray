mod camera;
mod context;
mod util;

pub use camera::*;
pub use context::RayGraphicsContext;

use crate::{report_depth_exhausted, Ray, Tracer};
use cgmath::{Vector3, Zero};

impl Ray {
    pub fn get_color<T: Tracer>(
        &self,
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        // Discard rays that have exhausted the depth limit
        if self.depth == 0 {
            report_depth_exhausted!(Vector3::zero(), "defaulting ray color to black")
        } else {
            match ctx.scene.cast_ray(self, &ctx.world) {
                Some((obj_id, intersection)) => {
                    let material = ctx.scene.get_object_material(obj_id);
                    let (obj, transform) = ctx.scene.get_object(obj_id.0).unwrap();
                    material.interact(self, &intersection, obj, transform, ctx, tracer, trace)
                }
                None => ctx.sky_color.clone(),
            }
        }
    }
}
