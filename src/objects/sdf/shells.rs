//! Geometries with a cutout of the same shape

use cgmath::Vector3;

use super::{
    sdf2d::{LineSegmentsRotationallyClosed, RotationallyClosedSDF},
    AABox, RescaleAbsoluteUniform, SDFObject,
};

/// A cone with a conical cutout in the center of the bottom plane. Both the radius and length of
/// the base cone and the cutout must be specified.
///
/// Without any transformation this object is centered about the origin and the
/// cone points in the positive Y-direction.
#[derive(Clone)]
pub struct ConeShell {
    outer_radius: f64,
    outer_length: f64,
    inner_radius: f64,
    inner_length: f64,
    inner_obj: LineSegmentsRotationallyClosed,
}

#[derive(Debug)]
pub enum ShellGeometryError {
    /// The parameters for the cutout section are too large for the base geometry
    CutoutTooLarge,
}

impl ConeShell {
    pub fn new(
        outer_radius: f64,
        outer_length: f64,
        inner_radius: f64,
        inner_length: f64,
    ) -> Result<Self, ShellGeometryError> {
        // Validate parameters
        if inner_radius >= outer_radius || inner_length >= outer_length {
            return Err(ShellGeometryError::CutoutTooLarge);
        }
        // Create geometry as a rotational body of a 2d path
        let inner_obj = LineSegmentsRotationallyClosed::new(
            -outer_length / 2. + inner_length,
            vec![
                (inner_radius, -outer_length / 2.).into(),
                (outer_radius, -outer_length / 2.).into(),
            ],
            outer_length / 2.,
        )
        .unwrap();
        Ok(Self {
            outer_radius,
            outer_length,
            inner_radius,
            inner_length,
            inner_obj,
        })
    }
}

impl SDFObject for ConeShell {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        SDFObject::sdf(&self.inner_obj as &dyn RotationallyClosedSDF, position)
    }

    fn sdf_grad(&self, position: &Vector3<f64>) -> Vector3<f64> {
        SDFObject::sdf_grad(&self.inner_obj as &dyn RotationallyClosedSDF, position)
    }

    fn bounding_box(&self) -> AABox {
        SDFObject::bounding_box(&self.inner_obj as &dyn RotationallyClosedSDF)
    }
}

impl RescaleAbsoluteUniform for ConeShell {
    fn rescale_absolute(&mut self, size_change: f64) {
        self.outer_radius += size_change / 2.;
        self.outer_length += size_change;
        self.inner_radius += size_change / 2.;
        self.inner_length += size_change;
        self.inner_obj = LineSegmentsRotationallyClosed::new(
            -self.outer_length / 2. + self.inner_length,
            vec![
                (self.inner_radius, -self.outer_length / 2.).into(),
                (self.outer_radius, -self.outer_length / 2.).into(),
            ],
            self.outer_length / 2.,
        )
        .unwrap();
    }
}
