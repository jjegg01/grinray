//! Contextual information used in raytracing with a graphical output

use cgmath::Vector3;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::{Scene, Tracer, World};

/// Context object for probabilistic raytracing in a geometric scene of objects
pub struct RayGraphicsContext<'a, T: Tracer> {
    pub(crate) scene: &'a Scene<T>,
    pub(crate) sky_color: Vector3<f32>,
    pub(crate) world: World,
    pub(crate) rng: Xoshiro256Plus,
}

impl<'a, T: Tracer> RayGraphicsContext<'a, T> {
    /// Create a new context with the given parameters
    pub fn new(scene: &'a Scene<T>, sky_color: Vector3<f32>, world: World, rng: Xoshiro256Plus) -> Self {
        Self {
            scene,
            sky_color,
            world,
            rng,
        }
    }

    /// Create a new context with:
    /// - A white sky color
    /// - The default world
    /// - An RNG initialized from system entropy
    pub fn with_defaults(scene: &'a Scene<T>) -> Self {
        Self {
            scene,
            sky_color: Vector3::new(1.0, 1.0, 1.0),
            world: World::default(),
            rng: Xoshiro256Plus::from_entropy()
        }
    }
}
