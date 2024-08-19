//! Hardcoded constants for GRINRAY

/// Minimum distance at which a ray intersection is detected
/// Note: Bidirectional "ray"tracing (i.e. the `intersect_line` function) is not affected by this
/// lower bound
pub(crate) const RAYDIST_EPSILON: f64 = 1e-7;
/// Casting rays over extreme distances can introduce large numerical errors. To prevent these
/// errors from causing problems down the line, we allow the raytracer to consider all intersections
/// with a distance larger than this value to be invalid (i.e. the same as "no intersection")
pub(crate) const RAYDIST_MAX: f64 = 1e10;
