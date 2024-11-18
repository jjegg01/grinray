use std::{path::PathBuf, process::ExitCode};

use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, Zero};
use clap::{Parser, ValueEnum};

use grinray::{
    graphics::{
        Camera, OrthographicCamaraParameters, OrthographicCamera, PerspectiveCamera,
        PerspectiveCameraParameters, RayGraphicsContext,
    },
    objects::{sdf, Cuboid, Cylinder, Hemisphere, ObjectTransform, Plane, RTObject, Sphere},
    CheckerboardMaterial, DebuggerTracer, FresnelMaterial, LambertMaterial,
    LinearGRINFresnelMaterial, Material, Scene, Tracer, World,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Render a simple scene of a particle on a plane with a checkerboard pattern
#[derive(Parser)]
struct Args {
    /// What particle to render
    #[clap(long, default_value = "sphere")]
    shape: ParticleShape,
    /// Material of the particle
    #[clap(long, default_value = "transparent")]
    material: ParticleMaterial,
    /// Strength of the refractive index gradient
    #[clap(long, default_value = "0")]
    grin_strength: f64,
    /// Refractive index of the particle (for non-zero GRIN strength, this is the refractive index
    /// at the reference point of the particle geometry)
    #[clap(long, default_value = "1.4")]
    base_index: f64,
    /// Rounding radius to apply to the edges of the geometry (note: ignored for non-SDF geometries)
    #[clap(long, default_value = "0")]
    rounding: f64,
    /// Camera projection to use
    #[clap(long, default_value = "perspective")]
    projection: Projection,
    /// Path of the output image
    output_image: PathBuf,
}

#[derive(ValueEnum, Clone)]
enum ParticleShape {
    Sphere,
    Cube,
    Cylinder,
    Hemisphere,
    SDFSphere,
    SDFCube,
    SDFCylinder,
    SDFCapsule,
    SDFCone
}

#[derive(ValueEnum, Clone)]
enum ParticleMaterial {
    Transparent,
    Opaque,
}

#[derive(ValueEnum, Clone)]
enum Projection {
    Orthographic,
    Perspective,
}

fn main() -> ExitCode {
    // Validate args
    let args = Args::parse();
    if let ParticleMaterial::Opaque = args.material {
        if args.grin_strength != 0. {
            eprintln!("WARNING: setting grin-strength for an opaque particle has no effect")
        }
    }
    if args.rounding < 0. {
        eprintln!("ERROR: negative rounding radii are invalid");
        return ExitCode::FAILURE;
    }
    // Macro for applying rounding operation to SDF geometries
    macro_rules! apply_rounding {
        ($inner: expr) => {
            if args.rounding > 0. {
                Box::new(sdf::Rounding::new_autoscale($inner, args.rounding))
            }
            else {
                Box::new($inner)
            }
        };
    }
    // Setup objects and their transformations
    let plane = Plane::new();
    let plane_transform = ObjectTransform::with_translation((0.0, -1.001, 0.0).into());
    let particle: Box<dyn RTObject + Send + Sync> = match args.shape {
        ParticleShape::Sphere => Box::new(Sphere::new(1.0)),
        ParticleShape::Cube => Box::new(Cuboid::new(2.0, 2.0, 2.0)),
        ParticleShape::Cylinder => Box::new(Cylinder::new(1.0, 2.0)),
        ParticleShape::Hemisphere => Box::new(Hemisphere::new(1.0)),
        ParticleShape::SDFSphere => apply_rounding!(sdf::Sphere::new(1.0)),
        ParticleShape::SDFCube => apply_rounding!(sdf::Cuboid::new(2.0, 2.0, 2.0)),
        ParticleShape::SDFCylinder => apply_rounding!(sdf::Cylinder::new(1.0, 2.0)),
        ParticleShape::SDFCapsule => apply_rounding!(sdf::Capsule::new(1.0, 1.0)),
        ParticleShape::SDFCone => apply_rounding!(sdf::Cone::new(1.0, 2.0)),
    };
    let particle_transform = match args.shape {
        ParticleShape::Sphere | ParticleShape::SDFSphere => {
            ObjectTransform::with_translation((0.0, 0.0, -3.0).into())
        }
        ParticleShape::Cube | ParticleShape::SDFCube => ObjectTransform::with_translation((0.0, 0.1, -4.0).into()),
        ParticleShape::Cylinder | ParticleShape::SDFCylinder | ParticleShape::SDFCone => ObjectTransform::with_translation((0.0, 0.1, -4.).into()),
        ParticleShape::Hemisphere => ObjectTransform::new(
            Quaternion::from_axis_angle(Vector3::new(1., 0., -1.).normalize(), Deg(-135.)),
            (0.0, 0.0, -3.0).into(),
        ),
        ParticleShape::SDFCapsule => ObjectTransform::new(Quaternion::from_angle_z(Deg(-45.)), (0.0, 0.6, -4.5).into())
    };
    // Setup materials
    let plane_mat = CheckerboardMaterial::new((1.0, 1.0, 1.0).into(), Vector3::unit_x());
    let particle_mat: Box<dyn Material<DebuggerTracer> + Send + Sync> = match args.material {
        ParticleMaterial::Transparent => {
            if args.grin_strength.is_zero() {
                Box::new(FresnelMaterial::new(args.base_index))
            } else {
                Box::new(LinearGRINFresnelMaterial::new(
                    args.base_index,
                    (0.0, args.grin_strength, 0.0).into(),
                ))
            }
        }
        ParticleMaterial::Opaque => Box::new(LambertMaterial::new(Vector3::new(1.0, 1.0, 1.0))),
    };
    // let particle_mat = LinearGRINFresnelMaterial::new(1.4, (0.0, 0.1, 0.0).into());
    // Setup scene
    let mut tracer = DebuggerTracer::new();
    let mut scene: Scene<DebuggerTracer> = Scene::new();
    let plane_mat = scene.add_material(Box::new(plane_mat));
    let particle_mat = scene.add_material(particle_mat);
    scene.add_object(Box::new(plane), plane_transform, plane_mat);
    scene.add_object(particle, particle_transform, particle_mat);
    // Create camera
    let camera: Box<dyn Camera<DebuggerTracer>> = match args.projection {
        Projection::Orthographic => {
            Box::new(OrthographicCamera::new(OrthographicCamaraParameters {
                pixels: (1024, 1024),
                samples: 32,
                ..Default::default()
            }))
        }
        Projection::Perspective => Box::new(PerspectiveCamera::new(PerspectiveCameraParameters {
            pixels: (1024, 1024),
            samples: 32,
            fov: (0.5f64).atan().to_degrees(),
            ..Default::default()
        })),
    };
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
    let cast_slice = bytemuck::cast_slice(buf.as_slice());
    let mut img = image::RgbaImage::from_vec(1024, 1024, cast_slice.to_vec()).unwrap();
    image::imageops::flip_vertical_in_place(&mut img);
    img.save(args.output_image).expect("Failed to save image");
    ExitCode::SUCCESS
}
