//! Material for rendering

use std::f64::consts::PI;

use cgmath::{ElementWise, InnerSpace, Matrix3, MetricSpace, SquareMatrix, Vector2, Vector3, Zero};
use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    ray::{RTIntersection, Ray, RayGraphicsContext},
    scene::RTObject,
    TraceEvent, Tracer,
};

/// Lower precision from Vector3<f64> to Vector3<f32>
fn vec3_64_to_32(v: &Vector3<f64>) -> Vector3<f32> {
    Vector3 {
        x: v.x as f32,
        y: v.y as f32,
        z: v.z as f32,
    }
}

/// Calculate next ray for the interaction between a Lambertian diffusor and a ray
fn lambert_next_ray(
    intersection: &RTIntersection,
    initial_depth: usize,
    rng: &mut Xoshiro256Plus,
) -> Option<Ray> {
    if initial_depth == 0 {
        return None;
    }
    let distr = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let unit_random = Vector3::new(distr.sample(rng), distr.sample(rng), distr.sample(rng));
    Some(Ray {
        start: intersection.point,
        dir: (intersection.normal + unit_random).normalize(),
        depth: initial_depth,
    })
}

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

/// Simple material that is a perfect Lambertian diffusor
#[derive(Clone)]
pub struct SimpleMaterial {
    /// Color tint of the material (multiplicative)
    color: Vector3<f32>,
}

impl SimpleMaterial {
    /// Create a new simple material with a solid color (color is multiplied with ray color during
    /// raytracing)
    pub fn new(color: Vector3<f32>) -> Self {
        Self { color }
    }
}

impl<T: Tracer> Material<T> for SimpleMaterial {
    fn interact(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        let next_ray = <SimpleMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            &mut ctx.rng,
            tracer,
            trace,
        );
        match next_ray {
            Some(next_ray) => self
                .color
                .mul_element_wise(next_ray.get_color(ctx, tracer, trace)),
            None => Vector3::zero(),
        }
    }

    fn next_ray(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        _: &(dyn RTObject + Send + Sync),
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
        lambert_next_ray(intersection, ray.depth - 1, rng)
    }
}

/// Checkerboard material that is mainly intended for the base plane so you can see refraction more nicely
#[derive(Clone)]
pub struct CheckerboardMaterial {
    color: Vector3<f32>,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

impl CheckerboardMaterial {
    /// Create a new material showing a checkerboard pattern.
    /// The pattern requires an origin, a direction and a color if you don't want white tiles
    pub fn new(color: Vector3<f32>, origin: Vector3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            color,
            origin,
            direction,
        }
    }
}

impl<T: Tracer> Material<T> for CheckerboardMaterial {
    fn interact(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        object: &(dyn RTObject + Send + Sync),
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        let next_ray = <CheckerboardMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
            &mut ctx.rng,
            tracer,
            trace,
        );
        match next_ray {
            Some(next_ray) => self
                .color
                .mul_element_wise(next_ray.get_color(ctx, tracer, trace)),
            None => Vector3::zero(),
        }
    }

    fn next_ray(
        &self,
        ray: &Ray,
        intersection: &RTIntersection,
        _: &(dyn RTObject + Send + Sync),
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Calculate checkerboard base color
        let point = vec3_64_to_32(&intersection.point);
        let normal = vec3_64_to_32(&intersection.normal);
        let plane_vec = self.origin - point;
        let direction2 = self.direction.cross(normal);
        let tex_x = plane_vec.dot(self.direction);
        let tex_y = plane_vec.dot(direction2);
        let sign_x = if tex_x < 0.0 { 1 } else { 0 };
        let sign_y = if tex_y < 0.0 { 1 } else { 0 };
        if ((tex_x as i32).abs() + (tex_y as i32).abs() + sign_x + sign_y) % 2 == 1 {
            // Diffuse reflection on "white" tiles
            tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
            lambert_next_ray(intersection, ray.depth - 1, rng)
        } else {
            // No exit ray on "black" tiles
            tracer.add_point(trace, TraceEvent::End, intersection.point);
            None
        }
    }
}

/// Perfectly smooth Fresnel refractor
#[derive(Clone)]
pub struct FresnelMaterial {
    /// Index inside the material
    index: f64,
    /// Index outside of the material
    outer_index: f64,
}

enum FresnelInteractionType {
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

    fn fresnel_interaction(
        index: f64,
        outer_index: f64,
        ray: &Ray,
        intersection: &RTIntersection,
        rng: &mut Xoshiro256Plus,
    ) -> FresnelInteractionType {
        // Get refractive indices on the incoming and outgoing side of the interface
        // as well as "canonical" normal vector that points towards the incoming ray
        let (n_in, n_out, normal) = if ray.dir.dot(intersection.normal) < 0. {
            // Ray entering object, i.e. surface normal points towards incoming ray
            (outer_index, index, intersection.normal)
        } else {
            // Ray leaving object
            (index, outer_index, -intersection.normal)
        };
        // -- Calculate reflected and transmitted ray intensities --
        let l_dot_n = ray.dir.dot(normal);
        let index_ratio = n_in / n_out;
        // Check for total reflection
        if index_ratio * index_ratio * (1. - l_dot_n * l_dot_n) > 1.0 {
            let dir = ray.dir - 2. * l_dot_n * normal;
            return FresnelInteractionType::Reflection(Ray {
                start: intersection.point,
                dir,
                depth: ray.depth - 1,
            });
        }
        let cosine_term = l_dot_n; // Dot product gives cosine of incident angle
        let sine_term = (1. - index_ratio * index_ratio * (1. - l_dot_n * l_dot_n)).sqrt(); // 1 - cos^2 = sin
                                                                                            // Fresnel equations
        let reflectance_s = ((index_ratio * (-cosine_term) - sine_term)
            / (index_ratio * (-cosine_term) + sine_term))
            .powf(2.);
        let reflectance_p = ((index_ratio * sine_term - (-cosine_term))
            / (index_ratio * sine_term + (-cosine_term)))
            .powf(2.);
        // TODO: track polarization
        let reflectance = (reflectance_p + reflectance_s) / 2.;
        // -- Monte-Carlo tracing --
        if rand::distributions::Uniform::new(0., 1.).sample(rng) < reflectance {
            // Reflection
            let dir = ray.dir - 2. * l_dot_n * normal;
            FresnelInteractionType::Reflection(Ray {
                start: intersection.point,
                dir,
                depth: ray.depth - 1,
            })
        } else {
            let dir = index_ratio * ray.dir - (index_ratio * cosine_term + sine_term) * normal;
            // Refraction
            FresnelInteractionType::Refraction(Ray {
                start: intersection.point,
                dir,
                depth: ray.depth - 1,
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
            match Self::fresnel_interaction(self.index, self.outer_index, &ray, &intersection, rng)
            {
                FresnelInteractionType::Reflection(next_ray) => {
                    tracer.add_point(trace, TraceEvent::Reflection, intersection.point);
                    // Inner reflections bounce on the inner surface
                    if inside {
                        ray = next_ray;
                        intersection = object.intersect_ray(&ray).unwrap();
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
                        intersection = object.intersect_ray(&ray).unwrap();
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

/// Fresnel material with a linear gradient index
#[derive(Clone)]
pub struct LinearGRINFresnelMaterial {
    /// Refractive index at the reference point
    reference_index: f64,
    gradient_strength: f64,
    gradient_dir: Vector3<f64>,
    outer_index: f64,
    /// Rotation to the GRIN reference frame where y-axis is the direction of the gradient
    reference_rotation: Matrix3<f64>,
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
    fn calc_reference_rotation(gradient_dir: Vector3<f64>) -> Matrix3<f64> {
        // Pre-calculate rotation matrix to align the gradient to the y-axis
        // If the gradient is already close to the y-axis, use the identity matrix
        // as the method below becomes unstable then
        if (gradient_dir.dot(Vector3::unit_y()) - 1.0).abs() < 1e-10 {
            Matrix3::identity()
        }
        // Otherwise calculate the vector perpendicular to both the gradient vector
        // and the y-axis and create a rotation matrix using this vector as the
        // axis of rotation
        else {
            let perp = gradient_dir.cross(Vector3::unit_y()).normalize();
            let angle = gradient_dir.angle(Vector3::unit_y());
            Matrix3::from_axis_angle(perp, angle)
        }
    }

    /// Create a new Fresnel material with a refractive index that varies linearly in space
    ///
    /// Requires the refractive index at the reference point of the geometric object you assign this
    /// material to, the gradient of the refractive index (i.e., direction and strength of the
    /// spatial variation) and the refractive index of the surrounding medium.
    pub fn new(reference_index: f64, gradient: Vector3<f64>, outer_index: f64) -> Self {
        let gradient_dir = gradient.normalize();
        let gradient_strength = gradient.magnitude();
        let reference_rotation = Self::calc_reference_rotation(gradient_dir);
        Self {
            reference_index,
            gradient_strength,
            gradient_dir,
            outer_index,
            reference_rotation,
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
        self.reference_rotation = Self::calc_reference_rotation(self.gradient_dir)
    }

    // Get refractive index at a specific point in space (needs reference point from geometry)
    fn index_at_point(&self, point: &Vector3<f64>, reference_point: &Vector3<f64>) -> f64 {
        self.reference_index
            + self.gradient_strength * self.gradient_dir.dot(point - reference_point)
    }

    // Analytical ray trajectory for position and tangent at a given arc length in GRIN materials
    // with the tangent tau0 as initial condition
    fn analytical_trajectory(&self, tau0: &Vector2<f64>, s: f64) -> (Vector2<f64>, Vector2<f64>) {
        let n01 = self.reference_index / self.gradient_strength;
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
        let n01 = self.reference_index / self.gradient_strength;
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
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Abort if ray has exhausted its depth counter
        if ray.depth == 0 {
            return None;
        }
        // Calculate rotation such that the ray direction is in the xy-plane
        // (after rotating such that the gradient is parallel to the y axis)
        let tmp = self.reference_rotation * ray.dir;
        let tmp = Vector2::new(tmp.x, tmp.z);
        let angle = tmp.angle(Vector2::unit_x());
        let ray_rotation = Matrix3::from_angle_y(-angle);
        // Full coordinate transform matrix
        let full_rotation = ray_rotation * self.reference_rotation;
        let full_rotation_inverse = full_rotation.invert().unwrap();
        // Transform ray to 2D coordinates
        let ray_2d_dir = full_rotation * ray.dir;
        assert!(ray_2d_dir.z.abs() < 1e-10); // DEBUG
        let tau0 = ray_2d_dir.truncate();
        // Loop until we arrive at a point that is close enough to the surface
        let mut trial_ray = ray.clone(); // Current point and direction along the ray trajectory
        let mut trial_s = 0.; // Current arclength distance along the trajectory
        let mut trial_tau_2d = tau0.clone(); // Current tangent vector *in the 2D frame*
        let exit_intersection = loop {
            // Intersect trial ray with object to get approximation for arc length
            let trial_intersection = match object.intersect_ray(&trial_ray) {
                Some(trial_intersection) => trial_intersection,
                None => object.intersect_line(&trial_ray).unwrap(),
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
                break trial_intersection;
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
        let exit_index = self.index_at_point(&exit_intersection.point, object.reference_point());
        match FresnelMaterial::fresnel_interaction(
            exit_index,
            self.outer_index,
            &ray,
            &exit_intersection,
            rng,
        ) {
            FresnelInteractionType::Reflection(reflected_ray) => {
                // On reflection at the inner surface we stay inside the material
                tracer.add_point(trace, TraceEvent::Reflection, reflected_ray.start);
                self.inner_trace(reflected_ray, object, rng, tracer, trace)
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
        ctx: &mut RayGraphicsContext<T>,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Vector3<f32> {
        // TODO: handle rays originating WITHIN the object (weird)
        if ray.dir.dot(intersection.normal) > 0. {
            unimplemented!("Ray may not start inside a GRIN material (yet)")
        }
        <LinearGRINFresnelMaterial as Material<T>>::next_ray(
            self,
            ray,
            intersection,
            object,
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
        rng: &mut Xoshiro256Plus,
        tracer: &mut T,
        trace: T::TraceID,
    ) -> Option<Ray> {
        // Calculate refractive index at the intersection point
        let index = self.index_at_point(&intersection.point, object.reference_point());
        // Calculate Fresnel interaction at the entry point
        match FresnelMaterial::fresnel_interaction(index, self.outer_index, ray, intersection, rng)
        {
            FresnelInteractionType::Reflection(next_ray) => {
                // Reflection is easy as the ray does not enter the material
                tracer.add_point(trace, TraceEvent::Reflection, next_ray.start);
                Some(next_ray)
            }
            FresnelInteractionType::Refraction(inner_ray) => {
                // Refractions takes us into the material
                tracer.add_point(trace, TraceEvent::Refraction, inner_ray.start);
                self.inner_trace(inner_ray, object, rng, tracer, trace)
            }
        }
    }
}
