use cgmath::Vector3;
use std::fmt::Debug;

/// Representation of a single light ray with a depth counter
#[derive(Clone)]
pub struct Ray {
    /// Start point of the ray
    pub start: Vector3<f64>,
    /// Ray direction
    pub dir: Vector3<f64>,
    /// Ray depth counter
    pub depth: usize,
}

/// Representation of an intersection between a light ray and an object
#[derive(Debug, Clone)]
pub struct RTIntersection {
    /// Distance from the ray origin to the intersection point
    pub ray_dist: f64,
    /// Point where the ray and the object intersect
    pub point: Vector3<f64>,
    /// Surface normal of the object at the intersection point
    pub normal: Vector3<f64>,
}

impl Debug for Ray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ray(start: ({}, {}, {}), dir: ({}, {}, {}))",
            self.start.x, self.start.y, self.start.z, self.dir.x, self.dir.y, self.dir.z
        )
    }
}
