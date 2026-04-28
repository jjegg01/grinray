//! Geometries with a cutout of the same shape

use cgmath::{InnerSpace, Vector2, Vector3};

use crate::objects::sdf::{Roundable, SURFACE_EPSILON, sdf2d::{MixedSegmentsRotationallyClosed, Segment}};

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

/// A hemisphere with a spherical cutout in the center of the bottom plane. Both the radius of the
/// base hemisphere and the cutout must be specified. The cut-plane is the XZ plane with the
/// hemisphere being in the space of positive Y coordinates.
///
/// Without any transformation the sphere center is at the origin.
#[derive(Clone)]
pub struct HemisphereShell {
    radius: f64,
    cutout_radius: f64,
    // Rescaling a hemisphere for rounding turns it into a non-hemisphere
    // We instead implement rounding manually
    rounding_radius: f64,
    inner: MixedSegmentsRotationallyClosed
}

impl HemisphereShell {
    pub fn new(radius: f64, cutout_radius: f64) -> Result<Self, ShellGeometryError> {
        if cutout_radius >= radius {
            return Err(ShellGeometryError::CutoutTooLarge)
        }
        // Create geometry as a rotational body of a 2d path
        let inner = MixedSegmentsRotationallyClosed::new(cutout_radius, vec![
            Segment::CircleArc { next_vertex: Vector2::new(cutout_radius, 0.), radius: -cutout_radius },
            Segment::Line { next_vertex: Vector2::new(radius, 0.) },
            Segment::CircleArc { next_vertex: Vector2::new(0., radius), radius }
        ]).unwrap();
        Ok(Self { radius, cutout_radius, inner, rounding_radius: 0.0 })
    }
}

impl SDFObject for HemisphereShell {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        SDFObject::sdf(&self.inner as &dyn RotationallyClosedSDF, position)
    }

    fn sdf_grad(&self, position: &Vector3<f64>) -> Vector3<f64> {
        SDFObject::sdf_grad(&self.inner as &dyn RotationallyClosedSDF, position)
    }

    fn bounding_box(&self) -> AABox {
        AABox {
            xlo: -self.radius - SURFACE_EPSILON * 1.1,
            xhi: self.radius + SURFACE_EPSILON * 1.1,
            ylo: 0. - SURFACE_EPSILON * 1.1,
            yhi: self.radius + SURFACE_EPSILON * 1.1,
            zlo: -self.radius - SURFACE_EPSILON * 1.1,
            zhi: self.radius + SURFACE_EPSILON * 1.1,
        }
    }
}

impl Roundable for HemisphereShell {
    type Rounded=Self;

    fn round(mut self, radius: f64) -> Self::Rounded {
        // Rounding is limited by thickness of shell
        self.rounding_radius += radius.min((self.radius - self.cutout_radius) / 2.);
        // Same as with hemisphere for the rounding of the outer edge
        let outer_rounding_start = Vector2::new(self.radius * (1. - 2. * self.rounding_radius / self.radius).sqrt(), 0.);
        let outer_rounding_end = Vector2::new(outer_rounding_start.x, self.rounding_radius).normalize() * self.radius;
        // Rounding points for the inner edge can be derived analogously to those of the outer edge
        let inner_rounding_end = Vector2::new(self.cutout_radius * (1. + 2. * self.rounding_radius / self.cutout_radius).sqrt(), 0.);
        let inner_rounding_start = Vector2::new(inner_rounding_end.x, self.rounding_radius).normalize() * self.cutout_radius;
        // Redefine 2d path
        self.inner = MixedSegmentsRotationallyClosed::new(self.cutout_radius, vec![
            Segment::CircleArc { next_vertex: inner_rounding_start, radius: -self.cutout_radius },
            Segment::CircleArc { next_vertex: inner_rounding_end, radius: self.rounding_radius},
            Segment::Line { next_vertex: outer_rounding_start },
            Segment::CircleArc { next_vertex: outer_rounding_end, radius: self.rounding_radius},
            Segment::CircleArc { next_vertex: Vector2::new(0., self.radius), radius: self.radius }
        ]).unwrap();
        self
    }
}