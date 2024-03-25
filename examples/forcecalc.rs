//! Example for using GRINRAY to calculate optical forces

use cgmath::{InnerSpace, Vector3, Zero};
use grinray::{LinearGRINFresnelMaterial, Material, RTObject, Ray, Sphere};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

#[allow(non_snake_case)]
fn main() {
    // Note: The values in this example are exaggerated for demonstrational purposes
    // We are using a length scale of micrometers here

    // Create a sphere at the origin with radius 10 and a refractive index varying from 1.3 to 1.5
    // on the z-axis
    const SPHERE_RADIUS: f64 = 10.;
    let particle = Sphere::new((0.,0.,0.).into(), SPHERE_RADIUS);
    const MEAN_INDEX: f64 = 1.52; // Typical polymer materials for refractive microparticles
    const GRADIENT_STRENGTH: f64 = 0.1;
    const MEDIUM_INDEX: f64 = 1.33; // Water
    let material = LinearGRINFresnelMaterial::new(MEAN_INDEX, (0.0, GRADIENT_STRENGTH / SPHERE_RADIUS, 0.0).into(), MEDIUM_INDEX);

    // Number of samples we take for every ray and maximum raytracing depth
    // (higher number => higher precision, but longer calculation)
    const SAMPLES_PER_RAY: usize = 32;
    const MAX_RAY_DEPTH: usize = 20;

    // Parameters for the light source (constant intensity parallel to z axis)
    const LIGHT_X_LO: f64 = -SPHERE_RADIUS;
    const LIGHT_X_HI: f64 = SPHERE_RADIUS;
    const LIGHT_Y_LO: f64 = -SPHERE_RADIUS;
    const LIGHT_Y_HI: f64 = SPHERE_RADIUS;
    const LIGHT_Z_START: f64 = -SPHERE_RADIUS - 1.;
    const LIGHT_INTENSITY: f64 = 1e7 * 1e-12; // 10^7 W/m^2 in W/µm^2
    const SPEED_OF_LIGHT: f64 = 299792458. / MEDIUM_INDEX; // m/s 

    // Step sizes for the spatial integration of the input light field
    const DX: f64 = 0.1;
    const DY: f64 = 0.1;
    const DA: f64 = DX * DY;
    let nx = ((LIGHT_X_HI - LIGHT_X_LO) / DX).round() as usize;
    let ny = ((LIGHT_Y_HI - LIGHT_Y_LO) / DX).round() as usize;

    // Seed the RNG
    let mut rng = Xoshiro256Plus::from_seed([
        161, 118, 62, 146, 170, 133, 194, 128, 48, 220, 173, 30, 87, 79, 187, 244,
        157, 55, 75, 236, 61, 31, 222, 232, 14, 233, 199, 29, 72, 25, 211, 184
    ]);

    // Perform the integration
    let mut force: Vector3<f64> = Vector3::zero();
    let mut torque: Vector3<f64> = Vector3::zero();
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
            if let Some(intersection) = particle.intersect_ray(&ray) {
                let mut successful_samples: usize = 0;
                // Accumulation buffers for force and torque caused by current ray
                let mut sample_force = Vector3::zero();
                let mut sample_torque = Vector3::zero();
                // Calculate incoming momentum and torque (normalized by ray intensity)
                let incoming_momentum = ray.dir.normalize();
                let incoming_angular_momentum = (intersection.point - particle.center).cross(incoming_momentum);
                for _ in 0..SAMPLES_PER_RAY {
                    if let Some(outgoing_ray) = material.next_ray(&ray, &intersection, &particle, &mut rng) {
                        // Calculate outgoing momentum and torque
                        let outgoing_momentum = outgoing_ray.dir.normalize();
                        let outgoing_angular_momentum = (outgoing_ray.start - particle.center).cross(outgoing_momentum);
                        sample_force += incoming_momentum - outgoing_momentum; // Units are fixed later
                        sample_torque += incoming_angular_momentum - outgoing_angular_momentum;
                        successful_samples += 1;
                    }
                }
                if successful_samples == 0 {
                    println!("Warning: failed to sample ray (try increasing the max ray depth)");
                }
                else {
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
    // the speed of light
    force *= DA * LIGHT_INTENSITY / SPEED_OF_LIGHT;
    torque *= DA * LIGHT_INTENSITY / SPEED_OF_LIGHT;

    // Scale units more sensibly
    force *= 1e12; // N -> pN
    torque *= 1e12; // N * µm -> pN * µm

    // Print output
    println!("Force: ({:.2}, {:.2}, {:.2}) pN", force.x, force.y, force.z);
    println!("Torque: ({:.2}, {:.2}, {:.2}) pN * µm", torque.x, torque.y, torque.z);
}