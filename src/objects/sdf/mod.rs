//! Objects defined by a signed distance function

mod sphere;

pub use sphere::*;

use cgmath::{InnerSpace, Rotation, Vector3};

use crate::{util::minmax, RTIntersection, Ray, World};

use super::{ObjectTransform, RTObject};

// Distance at which we consider an SDF object to be hit
const SURFACE_EPSILON: f64 = 1e-4;

pub struct AABox {
    pub xlo: f64,
    pub xhi: f64,
    pub ylo: f64,
    pub yhi: f64,
    pub zlo: f64,
    pub zhi: f64,
}

impl AABox {
    /// Helper function for determining intersections of the box with rays / lines
    fn ray_intersection_distance_bounds(&self, ray: &Ray) -> (f64, f64) {
        // This is basically a condensed version of the algorithm used for cuboids (see cuboid.rs)
        let s_upper = Vector3::new(
            (self.xhi - ray.start.x) / ray.dir.x,
            (self.yhi - ray.start.y) / ray.dir.y,
            (self.zhi - ray.start.z) / ray.dir.z,
        );
        let s_lower = Vector3::new(
            (self.xlo - ray.start.x) / ray.dir.x,
            (self.ylo - ray.start.y) / ray.dir.y,
            (self.zlo - ray.start.z) / ray.dir.z,
        );
        let (s_min_x, s_max_x) = minmax(s_lower.x, s_upper.x);
        let (s_min_y, s_max_y) = minmax(s_lower.y, s_upper.y);
        let (s_min_z, s_max_z) = minmax(s_lower.z, s_upper.z);
        let s_min = s_min_x.max(s_min_y).max(s_min_z);
        let s_max = s_max_x.min(s_max_y).min(s_max_z);
        (s_min, s_max)
    }

    /// Test if the given ray intersects this box
    fn ray_test(&self, ray: &Ray) -> Option<f64> {
        let (s_min, s_max) = self.ray_intersection_distance_bounds(ray);
        if s_max < s_min || s_max < 0. {
            None
        } else {
            Some(if s_min < 0. { s_max } else { s_min })
        }
    }

    /// Test if the given line intersects this box
    fn line_test(&self, ray: &Ray) -> Option<f64> {
        let (s_min, s_max) = self.ray_intersection_distance_bounds(ray);
        if s_max < s_min {
            None
        } else {
            Some(if s_min.abs() < s_max.abs() {
                s_min
            } else {
                s_max
            })
        }
    }

    fn is_inside(&self, point: &Vector3<f64>) -> bool {
        point.x >= self.xlo
            && point.x <= self.xhi
            && point.y >= self.ylo
            && point.y <= self.yhi
            && point.z >= self.zlo
            && point.z <= self.zhi
    }
}

pub trait SDFObject {
    /// Evaluate the signed distance function at a given point
    fn sdf(&self, position: &Vector3<f64>) -> f64;
    /// Gradient of the signed distance function (defaults to finite difference approximation)
    fn sdf_grad(&self, position: &Vector3<f64>) -> Vector3<f64> {
        let h = SURFACE_EPSILON;
        let fxf = self.sdf(&(position.x + h / 2., position.y, position.z).into());
        let fxb = self.sdf(&(position.x - h / 2., position.y, position.z).into());
        let fyf = self.sdf(&(position.x, position.y + h / 2., position.z).into());
        let fyb = self.sdf(&(position.x, position.y - h / 2., position.z).into());
        let fzf = self.sdf(&(position.x, position.y, position.z + h / 2.).into());
        let fzb = self.sdf(&(position.x, position.y, position.z - h / 2.).into());
        Vector3 {
            x: (fxf - fxb) / h,
            y: (fyf - fyb) / h,
            z: (fzf - fzb) / h,
        }
    }
    /// Bounding box of this object (i.e., the SDF must be negative outside this box)
    fn bounding_box(&self) -> AABox;
}

/// Helper function for SDF raytracing
/// When called with a position that has a distance within the surface epsilon from the SDF object,
/// it will advance this position along a ray until it either enters the object proper (i.e. SDF < 0)
/// or is definitely not inside the object (i.e. SDF > SURFACE_EPSILON). It also has a flag for
/// backwards marching (i.e., moving backwards along the given ray).
///
/// All parameters use object coordinates
fn pre_march<S: SDFObject>(
    obj: &S,
    current_ray_dist: &mut f64,
    current_position: &mut Vector3<f64>,
    current_surface_dist: &mut f64,
    ray: &Ray,
    backwards: bool,
) {
    // Since we are close to the object, we assume a flat surface in this estimate
    // For an incoming ray (i.e., ray direction is opposed to surface normal), we
    // expect a pre-marching distance of
    //   <current distance to surface> / cos(<angle between ray and surface normal>)
    // For outgoing rays we use the distance to the SDF = SURFACE_EPSILON surface instead
    // (note: this is flipped if we are doing backwards pre-marching)
    // We also limit the step size of the pre-marching to at most 10 * SURFACE_EPSILON so
    // that the assumption of a locally flat surface is actually upheld
    let normal_start_object_frame = obj.sdf_grad(current_position).normalize();
    let raydir_normal_dot = ray.dir.dot(normal_start_object_frame);
    let normal_dist = if (raydir_normal_dot < 0.) != backwards {
        *current_surface_dist
    } else {
        SURFACE_EPSILON - *current_surface_dist
    };
    let step_size = (1.1 * normal_dist / raydir_normal_dot.abs()).min(10. * SURFACE_EPSILON);
    loop {
        if backwards {
            *current_ray_dist -= step_size;
        } else {
            *current_ray_dist += step_size;
        }
        *current_position = ray.start + *current_ray_dist * ray.dir;
        *current_surface_dist = obj.sdf(current_position);
        if *current_surface_dist < 0. || *current_surface_dist > SURFACE_EPSILON {
            break;
        }
    }
}

enum MarchResult {
    /// Hit the objects surface
    Hit,
    /// Definitely misses the object
    Miss,
    /// Need to march further to tell
    Continue,
}

/// Helper function for SDF raytracing
///
/// All parameters use object coordinates
fn march_outside<S: SDFObject>(
    obj: &S,
    current_ray_dist: &mut f64,
    current_position: &mut Vector3<f64>,
    current_surface_dist: &mut f64,
    ray: &Ray,
    bbox: &AABox,
    backwards: bool,
) -> MarchResult {
    // March forward / backwards until we either leave the bounding box or get close enough to the surface
    if backwards {
        *current_ray_dist -= *current_surface_dist;
    } else {
        *current_ray_dist += *current_surface_dist;
    }
    *current_position = ray.start + *current_ray_dist * ray.dir;
    *current_surface_dist = obj.sdf(&current_position);
    // Did we leave the bounding box? Then no intersection
    if !bbox.is_inside(current_position) {
        return MarchResult::Miss;
    }
    // Did we get close enough to the surface for a hit?
    if *current_surface_dist < SURFACE_EPSILON {
        return MarchResult::Hit;
    }
    MarchResult::Continue
}

/// Helper function for SDF raytracing
///
/// All parameters use object coordinates
fn march_inside<S: SDFObject>(
    obj: &S,
    current_ray_dist: &mut f64,
    current_position: &mut Vector3<f64>,
    current_surface_dist: &mut f64,
    ray: &Ray,
    backwards: bool,
) -> MarchResult {
    // March forward / backwards until we get close enough to the surface
    // Note: Remember that the SDF is negative inside the object => negative sign marches forward
    if backwards {
        *current_ray_dist += SURFACE_EPSILON + *current_surface_dist;
    } else {
        *current_ray_dist += SURFACE_EPSILON - *current_surface_dist;
    }
    *current_position = ray.start + *current_ray_dist * ray.dir;
    *current_surface_dist = obj.sdf(current_position);
    // // Did we leave the bounding box? Then no intersection
    // if !bbox.is_inside(&current_position_obj_frame) {
    //     return None;
    // }
    // Trace until we have a positive surface distance (i.e., we exited the object)
    if *current_surface_dist > 0. {
        MarchResult::Hit
    } else {
        MarchResult::Continue
    }
}

/// Helper function for SDF raytracing
fn intersection_sdf<S: SDFObject>(obj: &S, ray_dist: &f64, point_obj_frame: &Vector3<f64>, transform: &ObjectTransform) -> RTIntersection {
    let gradient_object_frame = obj.sdf_grad(point_obj_frame).normalize();
    let normal = transform.rotation.rotate_vector(gradient_object_frame);
    let intersection_point = transform.point_to_world_frame(point_obj_frame);
    RTIntersection {
        ray_dist: *ray_dist,
        point: intersection_point,
        normal,
    }
}

impl<S: SDFObject> RTObject for S {
    fn intersect_ray(
        &self,
        transform: &super::ObjectTransform,
        ray: &Ray,
        _: &World
    ) -> Option<RTIntersection> {
        // Check if ray intersects the bounding box in object frame or is inside bbox
        let ray_object_frame = transform.ray_to_object_frame(ray);
        let bbox = self.bounding_box();
        let intitial_ray_dist = if bbox.is_inside(&ray_object_frame.start) {
            0.0
        } else {
            bbox.ray_test(&ray_object_frame)?
        };
        // Initialize current ray marching distance and position
        let mut current_ray_dist = intitial_ray_dist;
        let mut current_position_obj_frame =
            ray_object_frame.start + current_ray_dist * ray_object_frame.dir;
        // Are we inside the object?
        // This is a surprisingly difficult question to answer reliably, since the surface of an
        // object is a bit "fuzzy" in ray marching algorithms: we can only get close to the
        // surface but - at least in general - have no guarantee of hitting it exactly
        let mut current_surface_distance = self.sdf(&current_position_obj_frame);
        let outside = if current_surface_distance < 0. || current_surface_distance > SURFACE_EPSILON
        {
            // Easy cases:
            // - SDF negative => ray starts inside the object and is going out (eventually)
            // - SDF larger than SURFACE_EPSILON => ray starts outside the object and is coming in
            current_surface_distance > 0.
        } else {
            // Hard case:
            // In the thin layer around the true surface of the object we cannot use the starting
            // distance alone as an indicator whether the ray is inside or not since both
            // refracted and reflected rays start in this area.
            // Therefore, we do some "pre-marching" to leave the indeterminate region first
            pre_march(
                self,
                &mut current_ray_dist,
                &mut current_position_obj_frame,
                &mut current_surface_distance,
                &ray_object_frame,
                false,
            );
            current_surface_distance > 0.
        };
        // Tracing of incoming rays (i.e., rays that start outside the object)
        if outside {
            loop {
                match march_outside(
                    self,
                    &mut current_ray_dist,
                    &mut current_position_obj_frame,
                    &mut current_surface_distance,
                    &ray_object_frame,
                    &bbox,
                    false,
                ) {
                    MarchResult::Hit => break,
                    MarchResult::Miss => return None,
                    MarchResult::Continue => {}
                }
            }
        }
        // Tracing of outgoing rays (i.e., rays that start inside the object)
        else {
            loop {
                match march_inside(
                    self,
                    &mut current_ray_dist,
                    &mut current_position_obj_frame,
                    &mut current_surface_distance,
                    &ray_object_frame,
                    false,
                ) {
                    MarchResult::Hit => break,
                    MarchResult::Miss => {
                        panic!("Internal ray lost in SDF raytracing")
                    }
                    MarchResult::Continue => {}
                }
            }
        };
        let (intersection_ray_dist, intersection_point_obj_frame) =
            (current_ray_dist, current_position_obj_frame);
        // Determine surface normal via gradient of SDF
        Some(intersection_sdf(self, &intersection_ray_dist, &intersection_point_obj_frame, transform))
    }

    fn intersect_line(
        &self,
        transform: &super::ObjectTransform,
        line: &Ray,
        _: &World
    ) -> Option<RTIntersection> {
        // Check if ray intersects the bounding box in object frame or is inside bbox
        let line_object_frame = transform.ray_to_object_frame(&line);
        let bbox = self.bounding_box();
        let intitial_ray_dist = if bbox.is_inside(&line_object_frame.start) {
            0.0
        } else {
            bbox.line_test(&line_object_frame)?
        };
        // We raytrace both forwards and backwards and need to keep track of three pieces of
        // information:
        // - The current position (both absolute and in terms of the ray distance)
        // - The absolute distance we travelled from the starting point (since we want to find the
        //   closest intersection)
        let mut forward_ray_dist = intitial_ray_dist;
        let mut forward_pos_object_frame =
            line_object_frame.start + forward_ray_dist * line_object_frame.dir;
        let mut forward_surface_distance = self.sdf(&forward_pos_object_frame);
        let mut backward_ray_dist = intitial_ray_dist;
        let mut backward_pos_object_frame =
            line_object_frame.start + backward_ray_dist * line_object_frame.dir;
        let mut backward_surface_distance = forward_surface_distance;
        // Are we tracing inside the object or outside the object. Note, that the answer might depend
        // on the direction of the tracing!
        let (forward_outside, backward_outside) =
            if forward_surface_distance < 0. || forward_surface_distance > SURFACE_EPSILON {
                // Easy case: definitely outside or inside based on the start point
                let outside = forward_surface_distance > 0.;
                (outside, outside)
            } else {
                // Unfortunate case: we start in the intermediate region
                // First, pre-march both forward and backwards, then judge the positions for both
                // directions
                pre_march(
                    self,
                    &mut forward_ray_dist,
                    &mut forward_pos_object_frame,
                    &mut forward_surface_distance,
                    &line_object_frame,
                    false,
                );
                pre_march(
                    self,
                    &mut backward_ray_dist,
                    &mut backward_pos_object_frame,
                    &mut backward_surface_distance,
                    &line_object_frame,
                    true,
                );
                (
                    forward_surface_distance > 0.,
                    backward_surface_distance > 0.,
                )
            };
        // March greedily in both directions until we find a hit
        let (intersection_ray_dist, intersection_position_obj_frame) = loop {
            // Calculate the distances we have travelled so far in each direction
            let forward_travel_dist = (intitial_ray_dist - forward_ray_dist).abs();
            let backward_travel_dist = (intitial_ray_dist - backward_ray_dist).abs();
            // We use infinities to denote a miss, so if both travel distances are infinity, we can stop
            if forward_travel_dist.is_infinite() && backward_travel_dist.is_infinite() {
                return None
            }
            // Then, select the next closest next possible marching candidate
            // (either forward or backward)
            let (current_ray_dist, current_position_obj_frame, current_surface_distance, outside, backwards) =
                if forward_travel_dist + forward_surface_distance.abs()
                    < backward_travel_dist + backward_surface_distance.abs()
                {
                    (
                        &mut forward_ray_dist,
                        &mut forward_pos_object_frame,
                        &mut forward_surface_distance,
                        forward_outside,
                        false
                    )
                } else {
                    (
                        &mut backward_ray_dist,
                        &mut backward_pos_object_frame,
                        &mut backward_surface_distance,
                        backward_outside,
                        true
                    )
                };
            // Perform one marching step and look at the result:
            // - If we get a hit, we can break from the loop as we have found the closest intersection
            // - If we miss, we set the travelled distance to infinity to denote
            // - Else, we need to continue marching
            if outside {
                match march_outside(
                    self,
                    current_ray_dist,
                    current_position_obj_frame,
                    current_surface_distance,
                    &line_object_frame,
                    &bbox,
                    backwards,
                ) {
                    MarchResult::Hit => break (current_ray_dist, current_position_obj_frame),
                    MarchResult::Miss => { *current_ray_dist = f64::INFINITY },
                    MarchResult::Continue => {}
                }
            } else {
                match march_inside(
                    self,
                    current_ray_dist,
                    current_position_obj_frame,
                    current_surface_distance,
                    &line_object_frame,
                    backwards,
                ) {
                    MarchResult::Hit => break (current_ray_dist, current_position_obj_frame),
                    MarchResult::Miss => {
                        panic!("Internal ray lost in SDF raytracing")
                    }
                    MarchResult::Continue => {}
                }
            }
        };
        Some(intersection_sdf(self, intersection_ray_dist, intersection_position_obj_frame, transform))
    }
}
