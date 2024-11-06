//! Minimalistic setup for generating graphical output that is useful for
//! debugging new objects types

use cgmath::{Vector3, Zero};
use grinray::{
    graphics::RayGraphicsContext, objects::{Cuboid, ObjectTransform, Plane}, CheckerboardMaterial, FullTracer, LambertMaterial, Ray, Scene, Tracer, World
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    // Setup objects and their transformations
    let plane = Plane::new();
    let plane_transform = ObjectTransform::with_translation((0.0, -1.0, 0.0).into());
    let particle = Cuboid::new(1.0, 1.0, 1.0);
    let particle_transform = ObjectTransform::with_translation((0.0, 0.0, -2.0).into());
    // Setup materials
    let plane_mat = CheckerboardMaterial::new((1.0, 1.0, 1.0).into(), Vector3::unit_x());
    let particle_mat = LambertMaterial::new((1.0, 1.0, 1.0).into());
    // Setup scene
    let mut tracer = FullTracer::new();
    let mut scene: Scene<FullTracer> = Scene::new();
    let plane_mat = scene.add_material(Box::new(plane_mat));
    let particle_mat = scene.add_material(Box::new(particle_mat));
    scene.add_object(Box::new(plane), plane_transform, plane_mat);
    scene.add_object(Box::new(particle), particle_transform, particle_mat);
    // Create ray graphics context
    let mut ctx = RayGraphicsContext::new(
        &scene,
        (1.0, 1.0, 1.0).into(),
        World::default(),
        Xoshiro256Plus::from_seed([
            9, 41, 26, 176, 113, 164, 141, 6, 251, 27, 52, 143, 10, 196, 76, 147, 99, 215, 103,
            223, 78, 137, 249, 101, 252, 6, 139, 184, 69, 177, 191, 211,
        ]),
    );
    // Create a ray and get its color
    let ray = Ray {
        start: Vector3::zero(),
        dir: -Vector3::unit_z(),
        depth: 10,
    };
    let trace = tracer.new_trace(ray.start);
    let color = ray.get_color(&mut ctx, &mut tracer, trace);
    dbg!(color);
    tracer.end_trace(trace);
    // Output ray path (skipping intermediates)
    let trace = &tracer.get_traces()[0];
    for point in trace {
        match point.event {
            grinray::TraceEvent::Intermediate => {}
            _ => {
                println!(
                    "{:?}@[{},{},{}]",
                    point.event, point.location.x, point.location.y, point.location.z
                );
            }
        }
    }
}
