/// Struct for representing global parameters of the raytracing process (i.e, the "world" your
/// scene lives in).
pub struct World {
    /// Refractive index of the world medium
    pub refractive_index: f64,
    /// Minimum value for ray distances used for intersection tests
    /// (see [intersect_ray](crate::objects::RTObject::intersect_ray))
    pub ray_distance_epsilon: f64,
    /// Casting rays over extreme distances can introduce large numerical errors. To prevent these
    /// errors from causing problems down the line, we allow the raytracer to consider all intersections
    /// with a distance larger than this value to be invalid (i.e. the same as "no intersection")
    /// 
    /// This value has to be larger than the largest possible ray path in the scene.
    pub ray_distance_max: f64,
    /// Epsilon value used for geometries defined via signed distance functions. Ray tracing for
    /// these kinds of objects is based on ray marching, i.e., approaching the true surface iteratively.
    /// This value indicates the distance at which the iteration stops and a hit is registered.
    /// 
    /// IMPORTANT NOTE: This epsilon value *must* be smaller than the smallest distance of any SDF
    /// object in the scene to any explicit geometry and smaller than half the distance between any
    /// two SDF objects!
    /// 
    /// In general, the choice of this value is a compromise between performance and accuracy. Setting
    /// the value too high will lead to "fuzzy" objects (i.e., an inconsistent surface). Setting this
    /// value too low will lower performance.
    pub ray_surface_epsilon: f64
}

impl Default for World {
    fn default() -> Self {
        Self {
            refractive_index: 1.0,
            ray_distance_epsilon: 1e-7,
            ray_distance_max: 1e10,
            ray_surface_epsilon: 1e-3
        }
    }
}
