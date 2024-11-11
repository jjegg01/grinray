use std::f64::consts::PI;

use cgmath::{InnerSpace, MetricSpace, Quaternion, Rotation, Rotation3, Vector2, Vector3, Zero};
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    graphics::RayGraphicsContext, objects::{ObjectTransform, RTObject}, report_depth_exhausted, report_lost_ray, unwrap_lost_ray, FresnelInteractionType, FresnelMaterial, RTIntersection, Ray, TraceEvent, Tracer, World
};

use super::Material;

/// Fresnel material with a linear gradient index
#[derive(Clone)]
pub struct LinearGRINFresnelMaterial {
    /// Refractive index at the origin of the object-local coordinate system
    origin_index: f64,
    /// The strength of the gradient (i.e., change in refractive index per distance unit)
    gradient_strength: f64,
    /// Direction of the gradient as a unit vector
    gradient_dir: Vector3<f64>,
}

// Lower limit for the ray direction component perpendicular to the material gradient.
// Rays that fall below this value (i.e., rays that are "too parallel" to the gradient)
// will be considered to travel in a straight line to avoid numerical problems with the
// anlytical trajectory.
const LINEAR_GRIN_RAYDIR_EPSILON: f64 = 1e-7;
// The linear GRIN ray tracing works by alternating between analytically calculating
// points on the curved trajectory of a ray in a linear GRIN material and points on the
// surface of the object in question. This value determines when these two results are
// close enough to finish tracing.
const LINEAR_GRIN_ERROR: f64 = 1e-4;
// In concave geometries, one can "miss" parts of the surface when only casting a straigt
// ray at the start of each curved path segment. We can mitigate this by limiting the
// distance we make in one step based on the amout of curvature of the ray. This value
// indicates the (approximate) maximum of
const LINEAR_GRIN_ANGULAR_DEVIATION_THRESHOLD: f64 = 2. * PI * 1e-2;

// NOTE: We use a special coordinate system for analytically predicting the ray path in
// GRIN materials with a linear gradient. Here, the y-axis is defined by the direction
// of the gradient (with higher indices towards the positive axis)
impl LinearGRINFresnelMaterial {
    fn calc_reference_rotation(gradient_dir: Vector3<f64>, transform: &ObjectTransform) -> Quaternion<f64> {
        // Calculate actual gradient direction (in lab frame)
        let actual_gradient_dir = transform.rotation.rotate_vector(gradient_dir);
        Quaternion::from_arc(actual_gradient_dir, Vector3::unit_y(), None)
    }

    /// Create a new Fresnel material with a refractive index that varies linearly in space
    ///
    /// Requires the refractive index at the reference point of the geometric object you assign this
    /// material to, the gradient of the refractive index (i.e., direction and strength of the
    /// spatial variation) and the refractive index of the surrounding medium.
    pub fn new(reference_index: f64, gradient: Vector3<f64>) -> Self {
        let gradient_dir = gradient.normalize();
        let gradient_strength = gradient.magnitude();
        Self {
            origin_index: reference_index,
            gradient_strength,
            gradient_dir
        }
    }

    /// Return the normalized direction of the gradient
    pub fn get_gradient_dir(&self) -> Vector3<f64> {
        self.gradient_dir
    }

    /// Set the direction and strength gradient of the material
    pub fn set_gradient(&mut self, gradient: Vector3<f64>) {
        self.gradient_dir = gradient.normalize();
        self.gradient_strength = gradient.magnitude();
    }

    // Get refractive index at a specific point in space (needs reference point from geometry to
    // locate the origin of the gradient profile)
    fn index_at_point(&self, point: &Vector3<f64>, origin_point: &Vector3<f64>) -> f64 {
        self.origin_index
            + self.gradient_strength * self.gradient_dir.dot(point - origin_point)
    }

    // Analytical ray trajectory for position and tangent at a given arc length in GRIN materials
    // with the tangent tau0 as initial condition
    fn analytical_trajectory(&self, tau0: &Vector2<f64>, s: f64) -> (Vector2<f64>, Vector2<f64>) {
        let n01 = self.origin_index / self.gradient_strength;
        debug_assert!(n01 >= 0.);
        // If ray is nearly parallel to gradient, approximate as linear
        // TODO: higher order expansion?
        if tau0.x.abs() < LINEAR_GRIN_RAYDIR_EPSILON {
            (tau0 * s, *tau0)
        } else {
            let x = tau0.x
                * n01
                * (((s * s + 2. * tau0.y * n01 * s + n01 * n01).sqrt() + tau0.y * n01 + s)
                    / (n01 + tau0.y * n01))
                    .ln();
            let y = (s * s + 2. * tau0.y * n01 * s + n01 * n01).sqrt() - n01;
            let taux = tau0.x * n01 / (s * s + 2. * tau0.y * n01 * s + n01 * n01).sqrt();
            let tauy = (tau0.y * n01 + s) / (s * s + 2. * tau0.y * n01 * s + n01 * n01).sqrt();
            (Vector2::new(x, y), Vector2::new(taux, tauy))
        }
    }

    // Analytical solution for the arclength at which a given angular deviation is surpassed.
    // Angular deviation is here measured by the dot product of the tangent vector at the start
    // of the trajectory (tau0) and the tangent vector along the trajectory. More precisely, we
    // determine the arclength s at which the cosine between the initial tangent vector tau0
    // and the tangent vector at arclength s falls below a given value.
    //
    // For a linear gradient, the angular deviation is a strictly monotonic function of the
    // arclength, so any tangent vector further along the trajectory will deviate even more.
    //
    // NOTE: There is a maximum achievable angular deviation for each initial tangent vector
    // since the ray cannot bend beyond the direction of the gradient. Angular deviations
    // larger than this value cannot be achieved at any (positiv) arclength, so this function
    // returns +inf in this case
    fn analytical_arclen_angular_deviation(
        &self,
        tau0: &Vector2<f64>,
        angular_deviation_threshold: f64,
    ) -> f64 {
        let n01 = self.origin_index / self.gradient_strength;
        let threshold = angular_deviation_threshold; // Shorthand
        if tau0.y.abs() >= (threshold - 1e-10) {
            f64::INFINITY
        } else {
            n01 * (1.0 - threshold * threshold)
                * (-tau0.y / (tau0.y * tau0.y - threshold * threshold)
                    + (threshold * threshold * tau0.x * tau0.x / (1. - threshold * threshold))
                        .sqrt()
                        / (tau0.y * tau0.y - threshold * threshold).abs())
        }
    }

    // Trace a ray until it (probabilistically) leaves the object
    fn inner_trace<T: Tracer>(
        &self,
        ray: Ray,
        object: &(dyn RTObject + Send + Sync),
        transform: &ObjectTransform,
        gradient_reference_rotation: Quaternion<f64>,
        world: &World,
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Abort if ray has exhausted its depth counter
        if ray.depth == 0 {
            report_depth_exhausted!(return None, "defaulting linear GRIN next ray to None")
        }
        // Calculate rotation such that the ray direction is in the xy-plane
        // (after rotating such that the gradient is parallel to the y-axis)
        let tmp = gradient_reference_rotation * ray.dir;
        let tmp = Vector2::new(tmp.x, tmp.z);
        let angle = tmp.angle(Vector2::unit_x());
        let ray_rotation = Quaternion::from_angle_y(-angle);
        // Full coordinate transform matrix
        let full_rotation = ray_rotation * gradient_reference_rotation;
        let full_rotation_inverse = full_rotation.conjugate();
        // Transform ray to 2D coordinates
        let ray_2d_dir = full_rotation * ray.dir;
        debug_assert!(ray_2d_dir.z.abs() < 1e-10); // DEBUG
        let tau0 = ray_2d_dir.truncate();
        // Loop until we arrive at a point that is close enough to the surface
        let mut trial_ray = ray.clone(); // Current point and direction along the ray trajectory
        let mut trial_s = 0.; // Current arclength distance along the trajectory
        let mut trial_tau_2d = tau0.clone(); // Current tangent vector *in the 2D frame*
        let (exit_intersection, exit_tau) = loop {
            // Intersect trial ray with object to get approximation for arc length
            // Note: This is a bit more intricate than it might look. We want to
            // move along the trajectory of the ray so we test first for an
            // intersection going forward. If that does not succeed, we also allow
            // going backwards (and ignore the minimum step length).
            let trial_intersection = match object.intersect_ray(transform, &trial_ray, &world) {
                Some(trial_intersection) => trial_intersection,
                None => unwrap_lost_ray!(object.intersect_line(transform, &trial_ray, &world),
                    "cannot find way back to object during GRIN tracing"),
            };
            // Move along trajectory by the length of the cast ray, but not too far
            // to avoid missing concave parts of the geometry
            let max_arclen_step = self.analytical_arclen_angular_deviation(
                &trial_tau_2d,
                LINEAR_GRIN_ANGULAR_DEVIATION_THRESHOLD.cos(),
            );
            trial_s += trial_intersection.ray_dist.min(max_arclen_step);
            // Calculate exact point and tangent (both are in 2D coordinates initially!)
            let trial_point;
            (trial_point, trial_tau_2d) = self.analytical_trajectory(&tau0, trial_s);
            let trial_point = full_rotation_inverse * trial_point.extend(0.) + ray.start;
            let trial_tau = full_rotation_inverse * trial_tau_2d.extend(0.);
            // If the exact point is close enough to the linear intersection, break loop
            let error = trial_point.distance(trial_intersection.point);
            if error < LINEAR_GRIN_ERROR {
                break (trial_intersection, trial_tau);
            }
            // Otherwise, we use the trial point as the start of our next attempt
            else {
                trial_ray = Ray {
                    start: trial_point,
                    dir: trial_tau.normalize(),
                    depth: ray.depth, // unused
                }
            }
        };
        // Use arclength to generate intermediate points for tracer
        // Note: we do create "duplicate" points at the entry and exit (refraction/reflection + intermediate)
        let mut tracer_s = 0.;
        let mut tracer_tau_2d = tau0;
        loop {
            // Calculate trace point
            let (point_2d, tangent_2d) = self.analytical_trajectory(&tau0, tracer_s);
            let trace_point = full_rotation_inverse * point_2d.extend(0.) + ray.start;
            tracer.add_point(trace, TraceEvent::Intermediate, trace_point);
            // Take a step along the trajectory
            let arclen_limit = self.analytical_arclen_angular_deviation(
                &tracer_tau_2d,
                LINEAR_GRIN_ANGULAR_DEVIATION_THRESHOLD.cos(),
            );
            if tracer_s + arclen_limit < trial_s {
                tracer_s += arclen_limit;
                tracer_tau_2d = tangent_2d;
            }
            // If the step takes us beyond the end of the actual ray path,
            // truncate at the exit intersection and stop tracing
            else {
                tracer.add_point(trace, TraceEvent::Intermediate, exit_intersection.point);
                break;
            }
        }

        // Interaction at the (potential) exit site
        let exit_index = self.index_at_point(&exit_intersection.point, &transform.translation);
        match FresnelMaterial::fresnel_interaction(
            exit_index,
            world.refractive_index,
            exit_tau,
            ray.depth,
            exit_intersection.point,
            exit_intersection.normal,
            rng,
        ) {
            FresnelInteractionType::Reflection(reflected_ray) => {
                // On reflection at the inner surface we stay inside the material
                tracer.add_point(trace, TraceEvent::Reflection, reflected_ray.start);
                self.inner_trace(reflected_ray, object, transform, gradient_reference_rotation, world, rng, tracer, trace)
            }
            FresnelInteractionType::Refraction(exit_ray) => {
                // Ray exiting the object
                tracer.add_point(trace, TraceEvent::Refraction, exit_ray.start);
                Some(exit_ray)
            }
        }
    }
}

impl<T: Tracer> Material<T> for LinearGRINFresnelMaterial {
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
        <LinearGRINFresnelMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            transform,
            &ctx.world,
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
        world: &World,
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // For now, rays originating WITHIN the object are an error and will be discarded
        // This situation can sometimes arise at sharp corners (e.g., the corners of a cube)
        if ray.dir.dot(intersection.normal) > 0. {
            report_lost_ray!(return None,
                "Discarding ray that appears to start inside the Fresnel material"
            )
        }
        // Calculate refractive index at the intersection point
        let index = self.index_at_point(&intersection.point, &transform.translation);
        // Calculate rotation that aligns the gradient direction with the y-axis
        let gradient_reference_rotation = Self::calc_reference_rotation(self.gradient_dir, transform);
        // Calculate Fresnel interaction at the entry point
        match FresnelMaterial::fresnel_interaction(index, world.refractive_index, ray.dir, ray.depth, intersection.point, intersection.normal, rng)
        {
            FresnelInteractionType::Reflection(next_ray) => {
                // Reflection is easy as the ray does not enter the material
                tracer.add_point(trace, TraceEvent::Reflection, next_ray.start);
                Some(next_ray)
            }
            FresnelInteractionType::Refraction(inner_ray) => {
                // Refractions takes us into the material
                tracer.add_point(trace, TraceEvent::Refraction, inner_ray.start);
                self.inner_trace(inner_ray, object, transform, gradient_reference_rotation, world, rng, tracer, trace)
            }
        }
    }
}
