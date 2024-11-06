//! Minimalistic setup for generating graphical output of a single sphere

use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3};
use grinray::{
    graphics::{Camera, PerspectiveCamera, PerspectiveCameraParameters, RayGraphicsContext},
    objects::{Hemisphere, ObjectTransform, Plane},
    CheckerboardMaterial, DebuggerTracer, LinearGRINFresnelMaterial, Scene, Tracer, World,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    // Setup objects and their transformations
    let plane = Plane::new();
    let plane_transform = ObjectTransform::with_translation((0.0, -1.0, 0.0).into());
    let particle = Hemisphere::new(1.0);
    let particle_transform = ObjectTransform::new(
        Quaternion::from_axis_angle(Vector3::new(1., 0., -1.).normalize(), Deg(-135.)),
        (0.0, 0.0, -2.0).into(),
    );
    // Setup materials
    let plane_mat = CheckerboardMaterial::new((1.0, 1.0, 1.0).into(), Vector3::unit_x());
    let particle_mat = LinearGRINFresnelMaterial::new(1.4, (0.0, 0.1, 0.0).into());
    // Setup scene
    let mut tracer = DebuggerTracer::new();
    let mut scene: Scene<DebuggerTracer> = Scene::new();
    let plane_mat = scene.add_material(Box::new(plane_mat));
    let particle_mat = scene.add_material(Box::new(particle_mat));
    scene.add_object(Box::new(plane), plane_transform, plane_mat);
    scene.add_object(Box::new(particle), particle_transform, particle_mat);
    // Create camera
    let camera = PerspectiveCamera::new(PerspectiveCameraParameters {
        pixels: (1024, 1024),
        samples: 32,
        max_depth: 20,
        ..Default::default()
    });
    // Create graphics context
    let mut ctx = RayGraphicsContext::new(
        &scene,
        (1.0, 1.0, 1.0).into(),
        World::default(),
        Xoshiro256Plus::from_seed([
            9, 41, 26, 176, 113, 164, 141, 6, 251, 27, 52, 143, 10, 196, 76, 147, 99, 215, 103,
            223, 78, 137, 249, 101, 252, 6, 139, 184, 69, 177, 191, 211,
        ]),
    );
    // Render scene
    let mut buf = vec![0u32; 1024 * 1024];
    camera.render(&mut ctx, buf.as_mut_slice(), &mut tracer);
    // Save image
    let mut img =
        image::RgbaImage::from_vec(1024, 1024, bytemuck::cast_slice(buf.as_slice()).to_vec())
            .unwrap();
    image::imageops::flip_vertical_in_place(&mut img);
    img.save("hemisphere.png").expect("Failed to save image");
}
