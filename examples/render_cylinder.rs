//! Minimalistic setup for generating graphical output of a single sphere

use cgmath::Vector3;
use grinray::{
    graphics::{Camera, PerspectiveCamera, PerspectiveCameraParameters, RayGraphicsContext}, objects::{Cylinder, ObjectTransform, Plane}, CheckerboardMaterial, DebuggerTracer, LinearGRINFresnelMaterial, Scene, Tracer
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    // Setup objects and their transformations
    let plane = Plane::new();
    let plane_transform = ObjectTransform::with_translation((0.0, -1.0, 0.0).into());
    let cylinder = Cylinder::new(1.0, 2.0);
    let transform = ObjectTransform::with_translation((0.0, 0.1, -3.).into());
    // Setup materials
    let checkerboard_mat = CheckerboardMaterial::new((1.0, 1.0, 1.0).into(), Vector3::unit_x());
    let object_mat = LinearGRINFresnelMaterial::new(1.4, (0.0,0.1,0.0).into(), 1.0);
    // Setup scene
    let mut tracer = DebuggerTracer::new();
    let mut scene: Scene<DebuggerTracer> = Scene::new();
    let checkerboard_mat = scene.add_material(Box::new(checkerboard_mat));
    let lambert_mat = scene.add_material(Box::new(object_mat));
    scene.add_object(Box::new(plane), plane_transform, checkerboard_mat);
    scene.add_object(Box::new(cylinder), transform, lambert_mat);
    // Create camera
    let camera = PerspectiveCamera::new(PerspectiveCameraParameters {
        pixels: (1024,1024),
        samples: 32,
        ..Default::default()
    });
    // Create graphics context
    let mut ctx = RayGraphicsContext::new(
        &scene,
        (1.0, 1.0, 1.0).into(),
        Xoshiro256Plus::from_seed([
            9, 41, 26, 176, 113, 164, 141, 6, 251, 27, 52, 143, 10, 196, 76, 147, 99, 215, 103,
            223, 78, 137, 249, 101, 252, 6, 139, 184, 69, 177, 191, 211,
        ]),
    );
    // Render scene
    let mut buf = vec![0u32;1024*1024];
    camera.render(&mut ctx, buf.as_mut_slice(), &mut tracer);
    // Save image
    let mut img = image::RgbaImage::from_vec(1024, 1024, bytemuck::cast_slice(buf.as_slice()).to_vec())
        .unwrap();
    image::imageops::flip_vertical_in_place(&mut img);
    img.save("cylinder.png").expect("Failed to save image");
}
