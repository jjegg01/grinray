use cgmath::{InnerSpace, Vector2, Vector3};

use crate::objects::sdf::{Roundable, SURFACE_EPSILON, sdf2d::{MixedSegmentsRotationallyClosed, Segment}};

use super::{sdf2d::RotationallyClosedSDF, AABox, SDFObject};

/// Hemisphere with a given radius. The cut-plane is the XZ plane with the
/// hemisphere being in the space of positive Y coordinates.
///
/// Without any transformation the sphere center is at the origin.
#[derive(Clone)]
pub struct Hemisphere {
    radius: f64,
    // Rescaling a hemisphere for rounding turns it into a non-hemisphere
    // We instead implement rounding manually
    rounding_radius: f64,
    inner: MixedSegmentsRotationallyClosed
}

impl Hemisphere {
    pub fn new(radius: f64) -> Self {
        // Create geometry as a rotational body of a 2d path
        let inner = MixedSegmentsRotationallyClosed::new(0., vec![
            Segment::Line { next_vertex: Vector2::new(radius, 0.) },
            Segment::CircleArc { next_vertex: Vector2::new(0., radius), radius }
        ]).unwrap();
        Self { radius, rounding_radius: 0., inner }
    }
}

impl SDFObject for Hemisphere {
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

impl Roundable for Hemisphere {
    type Rounded = Self;

    fn round(mut self, radius: f64) -> Self {
        // Rounding radius is limited by radius of hemisphere
        self.rounding_radius += radius.min(self.radius / 2.);
        // The start and end point of the circle arc can be determined by placing a sphere such
        // that it touches both the rounded and the flat part of the hemisphere. By realizing that
        // the center of this sphere must live on the line between the origin of the hemisphere and
        // the point there the imagined sphere touches the rounded part of the sphere one can then
        // derive the following expressions.
        let rounding_start = Vector2::new(self.radius * (1. - 2. * self.rounding_radius / self.radius).sqrt(), 0.);
        let rounding_end = Vector2::new(rounding_start.x, self.rounding_radius).normalize() * self.radius;
        self.inner = MixedSegmentsRotationallyClosed::new(0., vec![
            Segment::Line { next_vertex: rounding_start },
            Segment::CircleArc { next_vertex: rounding_end, radius:self.rounding_radius },
            Segment::CircleArc { next_vertex: Vector2::new(0., self.radius), radius: self.radius }
        ]).unwrap();
        self
    }
}

#[cfg(test)]
mod test {
    use cgmath::Vector3;

    use crate::objects::sdf::SDFObject;

    use super::Hemisphere;

    #[test]
    fn test_sdf() {
        let hemisphere = Hemisphere::new(1.0);
        assert!((hemisphere.sdf(&Vector3::new(0., 1., 0.)) - 0.0).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(0., 2., 0.)) - 1.0).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(0., 0.5, 0.)) + 0.5).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(1., 1., 0.)) - ((2.0f64).sqrt() - 1.)).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(2., 0., 0.)) - 1.0).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(1., -1., 0.)) - 1.0).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(0.5, -1., 0.)) - 1.0).abs() < 1e-10);
        assert!((hemisphere.sdf(&Vector3::new(0., -1., 0.)) - 1.0).abs() < 1e-10);
    }
}