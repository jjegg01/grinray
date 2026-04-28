//! Analytical formulas for the center-of-mass for various particle shapes

use cgmath::{Vector3, Zero};
use grinray::objects::ObjectTransform;

use crate::ParticleShapeParameters;

/// Calculates center of mass for different particle shapes
pub(crate) fn calc_center_of_mass(shape: &ParticleShapeParameters, transform: &ObjectTransform) -> Vector3<f64> {
    let com_untransformed = match shape {
        ParticleShapeParameters::Sphere { .. } => Vector3::zero(),
        ParticleShapeParameters::Hemisphere { radius } => {
            Vector3::unit_y() * 3. / 8. * *radius
        },
        ParticleShapeParameters::HemisphereShell { radius, radius_cutout } => {
            Vector3::unit_y() * 3. / 8. * (radius.powi(4) - radius_cutout.powi(4)) / (radius.powi(3) - radius_cutout.powi(3))
        },
        ParticleShapeParameters::Cone { length, .. } => {
            Vector3::unit_y() * *length / 4.
        },
        ParticleShapeParameters::ConeShell { length, radius, length_cutout, radius_cutout } => {
            Vector3::unit_y() * 1. / 4. * (radius.powi(2) * length.powi(2) - radius_cutout.powi(2) * length_cutout.powi(2)) /
                (radius.powi(2) * length - radius_cutout.powi(2) * length_cutout)
        },
        ParticleShapeParameters::Cube { .. } => Vector3::zero(),
    };
    transform.point_to_world_frame(&com_untransformed)
}