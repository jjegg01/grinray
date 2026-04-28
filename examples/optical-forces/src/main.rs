use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
};

use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, Zero};
use clap::{Parser, Subcommand, ValueEnum};
use grinray::{
    objects::{sdf, ObjectTransform, RTObject},
    FresnelMaterial, FullTracer, LinearGRINFresnelMaterial, Material, Ray, Tracer,
    World,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::com::calc_center_of_mass;

mod com;

/// Demo program to show how GRINRAY can be used to calculate the force and torques acting on a
/// particle with a symmetry broken refractive index profile under illumination.
#[derive(Parser)]
struct Args {
    /// Refractive index of the particle material (for GRIN particles this is the refractive index at the reference location for the shape used)
    #[clap(long, default_value = "1.509")]
    particle_refractive_index: f64,
    /// Refractive index for the medium the particle is suspended in
    #[clap(long, default_value = "1.33")]
    medium_refractive_index: f64,
    /// Strength of the refractive index gradient
    #[clap(long, default_value = "0", allow_negative_numbers = true)]
    grin_strength: f64,
    /// Shape of the particle
    #[clap(long, default_value = "hemisphere")]
    shape: ParticleShape,
    /// Simulation action to perform
    #[clap(subcommand)]
    command: SimulationCommand,
}

#[derive(Parser, Clone)]
struct SingleArgs {
    #[clap(long, default_value = "0", allow_negative_numbers = true)]
    angle: f64,
}

#[derive(Parser, Clone)]
struct CrossSectionArgs {
    #[clap(long, default_value = "0", allow_negative_numbers = true)]
    angle: f64,
    #[clap(long, default_value = "128", allow_negative_numbers = true)]
    samples_per_position: usize,
    #[clap(long, default_value = "0.25")]
    sample_delta: f64,
    output_filename_prefix: String,
}

#[derive(Parser, Clone)]
struct ScanArgs {
    #[clap(long, default_value = "-90.001", allow_negative_numbers = true)]
    start_angle: f64,
    #[clap(long, default_value = "90.001", allow_negative_numbers = true)]
    end_angle: f64,
    #[clap(long, default_value = "181")]
    steps: usize,
    output_filename: PathBuf,
}

#[derive(Subcommand, Clone)]
enum SimulationCommand {
    /// Calculate the force and torque acting on a particle in a single orientation
    Single(SingleArgs),
    /// Calculate ray paths for a cross section of the particle (useful for visualizing the paths travelled by the light rays)
    CrossSection(CrossSectionArgs),
    /// Calculate the force and torque acting on a particle for all multiple orientations
    Scan(ScanArgs),
}

#[derive(ValueEnum, Clone, Copy)]
enum ParticleShape {
    Sphere,
    Hemisphere,
    HemisphereShell,
    Cone,
    ConeShell,
    Cube
}

// -- UNITS --
// All lengths are in µm

// Particle size parameters
const SPHERE_RADIUS: f64 = 5.;
const CUTOUT_RADIUS: f64 = 3.;

// Number of samples we take for every ray and maximum raytracing depth
// (higher number => higher precision, but longer calculation)
const SAMPLES_PER_RAY: usize = 32;
const MAX_RAY_DEPTH: usize = 30;

// Parameters for the light source (constant intensity parallel to z axis)
const LIGHT_X_LO: f64 = -SPHERE_RADIUS;
const LIGHT_X_HI: f64 = SPHERE_RADIUS;
const LIGHT_Y_LO: f64 = -SPHERE_RADIUS;
const LIGHT_Y_HI: f64 = SPHERE_RADIUS;
const LIGHT_Z_START: f64 = -SPHERE_RADIUS - 2.;
const LIGHT_INTENSITY: f64 = 1e7 * 1e-12; // 10^7 W/m^2 in W/µm^2

// Step sizes for the spatial integration of the input light field
const DX: f64 = 0.1;
const DY: f64 = 0.1;
const DA: f64 = DX * DY;

pub(crate) enum ParticleShapeParameters {
    Sphere {
        radius: f64,
    },
    Hemisphere {
        radius: f64,
    },
    HemisphereShell {
        radius: f64,
        radius_cutout: f64,
    },
    Cone {
        length: f64,
        radius: f64,
    },
    ConeShell {
        length: f64,
        radius: f64,
        length_cutout: f64,
        radius_cutout: f64,
    },
    Cube {
        edge_length: f64
    }
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Create particle

    let shape_parameters = match &args.shape {
        ParticleShape::Sphere => ParticleShapeParameters::Sphere {
            radius: SPHERE_RADIUS * 4. / 5.,
        },
        ParticleShape::Hemisphere => ParticleShapeParameters::Hemisphere {
            radius: SPHERE_RADIUS,
        },
        ParticleShape::HemisphereShell => ParticleShapeParameters::HemisphereShell {
            radius: SPHERE_RADIUS,
            radius_cutout: CUTOUT_RADIUS,
        },
        ParticleShape::Cone => ParticleShapeParameters::Cone {
            length: SPHERE_RADIUS,
            radius: SPHERE_RADIUS,
        },
        ParticleShape::ConeShell => ParticleShapeParameters::ConeShell {
            length: SPHERE_RADIUS,
            radius: SPHERE_RADIUS,
            length_cutout: CUTOUT_RADIUS,
            radius_cutout: CUTOUT_RADIUS,
        },
        ParticleShape::Cube => ParticleShapeParameters::Cube {
            edge_length: SPHERE_RADIUS
        },
    };
    let particle: Box<dyn RTObject + Send + Sync> = match &shape_parameters {
        ParticleShapeParameters::Sphere { radius } => Box::new(sdf::Sphere::new(*radius)),
        ParticleShapeParameters::Hemisphere { radius } => Box::new(sdf::Hemisphere::new(*radius)),
        ParticleShapeParameters::HemisphereShell {
            radius,
            radius_cutout,
        } => Box::new(sdf::HemisphereShell::new(*radius, *radius_cutout).unwrap()),
        ParticleShapeParameters::Cone { length, radius } => {
            Box::new(sdf::Cone::new(*radius, *length))
        }
        ParticleShapeParameters::ConeShell {
            length,
            radius,
            length_cutout,
            radius_cutout,
        } => {
            Box::new(sdf::ConeShell::new(*radius, *length, *radius_cutout, *length_cutout).unwrap())
        }
        ParticleShapeParameters::Cube { edge_length } => {
            Box::new(sdf::Cuboid::new(*edge_length, *edge_length, *edge_length))
        },
    };

    let world = World {
        refractive_index: args.medium_refractive_index,
        ..Default::default()
    };

    fn build_material<T: Tracer>(args: &Args) -> Box<dyn Material<T>> {
        match args.grin_strength {
            0.0 => Box::new(FresnelMaterial::new(args.particle_refractive_index)),
            grin_strength => Box::new(LinearGRINFresnelMaterial::new(
                args.particle_refractive_index,
                Vector3::unit_y() * grin_strength,
            )),
        }
    }

    match &args.command {
        // Calculate forces and torques for a single angle
        SimulationCommand::Single(single_args) => {
            let material = build_material::<()>(&args);
            let particle_transform =
                ObjectTransform::with_rotation(Quaternion::from_angle_x(Deg(single_args.angle)));
            let center_of_mass = calc_center_of_mass(&shape_parameters, &particle_transform);

            let simulation_result = calculate_propulsion::<()>(
                particle.as_ref(),
                &center_of_mass,
                &particle_transform,
                material.as_ref(),
                &world,
                &mut (),
            );

            print_simulation_statistics(&simulation_result);

            // Print output
            println!(
                "Force: ({:.5}, {:.5}, {:.5}) pN",
                simulation_result.force.x, simulation_result.force.y, simulation_result.force.z
            );
            println!(
                "Torque: ({:.5}, {:.5}, {:.5}) pN * µm",
                simulation_result.torque.x, simulation_result.torque.y, simulation_result.torque.z
            );
        }
        // Calculate sample ray paths
        SimulationCommand::CrossSection(crosssection_args) => {
            let material = build_material::<FullTracer>(&args);
            let particle_transform = ObjectTransform::with_rotation(Quaternion::from_angle_x(Deg(
                crosssection_args.angle,
            )));

            // Seed the RNG
            let mut rng = Xoshiro256Plus::from_seed([
                161, 118, 62, 146, 170, 133, 194, 128, 48, 220, 173, 30, 87, 79, 187, 244, 157, 55,
                75, 236, 61, 31, 222, 232, 14, 233, 199, 29, 72, 25, 211, 184,
            ]);

            // Cross section 1: slice along xz
            let mut tracer = FullTracer::new();
            let mut inout_rays = HashMap::new();
            let dx = crosssection_args.sample_delta;
            let nx = ((LIGHT_X_HI - LIGHT_X_LO) / dx).round() as usize;
            for ix in 0..nx {
                let xpos = LIGHT_X_LO + dx / 2. + dx * ix as f64;
                // Build ray
                let ray = Ray {
                    start: Vector3::new(xpos, 0., LIGHT_Z_START),
                    dir: Vector3::unit_z(),
                    depth: MAX_RAY_DEPTH,
                };
                // Only trace rays that actually intersect the particle
                if let Some(intersection) =
                    particle.intersect_ray(&particle_transform, &ray, &world)
                {
                    for _ in 0..crosssection_args.samples_per_position {
                        let trace = tracer.new_trace(ray.start);
                        if let Some(outgoing_ray) = material.next_ray(
                            &ray,
                            &intersection,
                            particle.as_ref(),
                            &particle_transform,
                            &world,
                            &mut rng,
                            &mut tracer,
                            trace,
                        ) {
                            inout_rays.insert(trace, (ray.clone(), outgoing_ray));
                        } else {
                            println!("LOST RAY")
                        }
                        tracer.end_trace(trace);
                    }
                }
            }

            let output_file = File::create(format!(
                "{}-xz.csv",
                crosssection_args.output_filename_prefix
            )).expect("Failed to create output file for xz slice");
            let mut file_writer = BufWriter::new(output_file);
            let traces = tracer.get_traces();
            for (trace_idx, (in_ray, out_ray)) in inout_rays {
                let trace_idx: usize = trace_idx.into();
                let trace = &traces[trace_idx];
                let trace: Vec<String> = trace
                    .into_iter()
                    .map(|point| {
                        format!(
                            "{},{},{}",
                            point.location.x, point.location.y, point.location.z
                        )
                    })
                    .collect();
                fn ray_to_string(ray: &Ray) -> String {
                    format!(
                        "{},{},{},{},{},{}",
                        ray.start.x, ray.start.y, ray.start.z, ray.dir.x, ray.dir.y, ray.dir.z
                    )
                }
                writeln!(
                    file_writer,
                    "{},{},{}",
                    ray_to_string(&in_ray),
                    trace.join(","),
                    ray_to_string(&out_ray)
                ).expect("Failed to write output line");
            }
        }
        // Calculate forces and torques for multiple angles all at once
        SimulationCommand::Scan(scan_args) => {
            let material = build_material::<()>(&args);
            let mut output_file =
                File::create(&scan_args.output_filename).expect("Failed to create output file");
            writeln!(output_file, "# Angle (deg), Force X (pN), Force Y (pN), Force Z (pN), Torque X (pN * µm), Torque Y (pN * µm), Torque Z (pN * µm)").expect("Failed to write header to output file");
            for i in 0..scan_args.steps {
                let angle = scan_args.start_angle
                    + (scan_args.end_angle - scan_args.start_angle) / (scan_args.steps - 1) as f64
                        * i as f64;
                let particle_transform =
                    ObjectTransform::with_rotation(Quaternion::from_angle_x(Deg(angle)));
                let center_of_mass = calc_center_of_mass(&shape_parameters, &particle_transform);

                println!("Simulating for angle {angle}°...");

                let simulation_result = calculate_propulsion::<()>(
                    particle.as_ref(),
                    &center_of_mass,
                    &particle_transform,
                    material.as_ref(),
                    &world,
                    &mut (),
                );

                print_simulation_statistics(&simulation_result);

                writeln!(
                    output_file,
                    "{angle},{},{},{},{},{},{}",
                    simulation_result.force.x,
                    simulation_result.force.y,
                    simulation_result.force.z,
                    simulation_result.torque.x,
                    simulation_result.torque.y,
                    simulation_result.torque.z,
                )
                .expect("Failed to write simulation result");
            }
        }
    }
    ExitCode::SUCCESS
}

struct SimulationResult {
    force: Vector3<f64>,
    torque: Vector3<f64>,
    total_rays: usize,
    total_rays_lost: usize,
}

fn print_simulation_statistics(simulation_result: &SimulationResult) {
    println!("Simulation statistics:");
    println!(
        "  Lost rays: {} / {} ({:.2} %)",
        simulation_result.total_rays_lost,
        simulation_result.total_rays,
        simulation_result.total_rays_lost as f64 / simulation_result.total_rays as f64 * 100.
    );
}

fn calculate_propulsion<T: Tracer>(
    particle: &(dyn RTObject + Send + Sync),
    center_of_mass: &Vector3<f64>,
    transform: &ObjectTransform,
    material: &dyn Material<T>,
    world: &World,
    tracer: &mut T,
) -> SimulationResult {
    let nx = ((LIGHT_X_HI - LIGHT_X_LO) / DX).round() as usize;
    let ny = ((LIGHT_Y_HI - LIGHT_Y_LO) / DX).round() as usize;

    // Seed the RNG
    let mut rng = Xoshiro256Plus::from_seed([
        161, 118, 62, 146, 170, 133, 194, 128, 48, 220, 173, 30, 87, 79, 187, 244, 157, 55, 75,
        236, 61, 31, 222, 232, 14, 233, 199, 29, 72, 25, 211, 184,
    ]);

    // Perform the integration
    let mut force: Vector3<f64> = Vector3::zero();
    let mut torque: Vector3<f64> = Vector3::zero();
    let mut total_rays: usize = 0;
    let mut total_rays_lost: usize = 0;
    for iy in 0..ny {
        // Sample y positions at the center of each area element
        let ypos = LIGHT_Y_LO + DY / 2. + DY * iy as f64;
        for ix in 0..nx {
            let xpos = LIGHT_X_LO + DX / 2. + DX * ix as f64;
            let ray = Ray {
                start: Vector3::new(xpos, ypos, LIGHT_Z_START),
                dir: Vector3::unit_z(),
                depth: MAX_RAY_DEPTH,
            };
            // Only trace rays that actually intersect the particle
            if let Some(intersection) = particle.intersect_ray(&transform, &ray, &world) {
                let mut successful_samples: usize = 0;
                // Accumulation buffers for force and torque caused by current ray
                let mut sample_force = Vector3::zero();
                let mut sample_torque = Vector3::zero();
                // Calculate incoming momentum and angular momentum (normalized by ray intensity)
                let incoming_momentum = ray.dir.normalize();
                let incoming_angular_momentum =
                    (intersection.point - center_of_mass).cross(incoming_momentum);
                for _ in 0..SAMPLES_PER_RAY {
                    let trace = tracer.new_trace(ray.start);
                    if let Some(outgoing_ray) = material.next_ray(
                        &ray,
                        &intersection,
                        particle,
                        &transform,
                        &world,
                        &mut rng,
                        tracer,
                        trace,
                    ) {
                        // Calculate outgoing momentum and torque
                        let outgoing_momentum = outgoing_ray.dir.normalize();
                        let outgoing_angular_momentum =
                            (outgoing_ray.start - center_of_mass).cross(outgoing_momentum);
                        sample_force += incoming_momentum - outgoing_momentum; // Units are fixed later
                        sample_torque += incoming_angular_momentum - outgoing_angular_momentum;
                        successful_samples += 1;
                    } else {
                        total_rays_lost += 1;
                    }
                    total_rays += 1;
                }
                if successful_samples == 0 {
                    println!("Warning: failed to sample ray starting at ({}, {}) (try increasing the max ray depth)", xpos, ypos);
                } else {
                    force += sample_force / successful_samples as f64;
                    torque += sample_torque / successful_samples as f64;
                }
            }
        }
    }

    // Fix units:
    // On full absorption, each ray causes a force
    // dF = dA * I / c
    // with dA being the area element of the light source, I being the light intensity and c being
    // the speed of light in the suspending medium
    let speed_of_light: f64 = 299792458. / world.refractive_index; // m/s
    force *= DA * LIGHT_INTENSITY / speed_of_light;
    torque *= DA * LIGHT_INTENSITY / speed_of_light;

    // Scale units more sensibly
    force *= 1e12; // N -> pN
    torque *= 1e12; // N * µm -> pN * µm

    SimulationResult {
        force,
        torque,
        total_rays,
        total_rays_lost,
    }
}
