use cgmath::Vector3;

use super::{sdf2d::{LineSegmentsRotationallyClosed, RotationallyClosedSDF}, AABox, RescaleAbsoluteUniform, SDFObject};

/// A cone with a given radius and length.
///
/// Without any transformation the origin of the cone is in the center of its base and the
/// tip of the cone points in the positive Y-direction.
#[derive(Clone)]
pub struct Cone {
    radius: f64,
    length: f64,
    inner: LineSegmentsRotationallyClosed
}

impl Cone {
    pub fn new(radius: f64, length: f64) -> Self {
        // Create geometry as a rotational body of a 2d path
        let inner = LineSegmentsRotationallyClosed::new(
            0., // Center of base plane
            vec![(radius, 0.).into()], // Rim of base plane
            length // Tip of cone
        ).unwrap();
        Self { radius, length, inner }
    }
}

impl SDFObject for Cone {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        SDFObject::sdf(&self.inner as &dyn RotationallyClosedSDF, position)
    }

    fn sdf_grad(&self, position: &Vector3<f64>) -> Vector3<f64> {
        SDFObject::sdf_grad(&self.inner as &dyn RotationallyClosedSDF, position)
    }

    fn bounding_box(&self) -> AABox {
        SDFObject::bounding_box(&self.inner as &dyn RotationallyClosedSDF)
    }
}

impl RescaleAbsoluteUniform for Cone {
    fn rescale_absolute(&mut self, size_change: f64) {
        self.radius += size_change / 2.;
        self.length += size_change;
        self.inner = LineSegmentsRotationallyClosed::new(0., vec![(self.radius, 0.).into()], self.length).unwrap();
    }
}

#[cfg(test)]
mod test {
    use cgmath::Vector3;

    use crate::objects::sdf::SDFObject;

    use super::Cone;

    #[test]
    fn test_sdf() {
        let cone = Cone::new(0.5, 1.0);
        assert!((cone.sdf(&Vector3::new(0.,  1.0, 0.))).abs() < 1e-10);
        assert!((cone.sdf(&Vector3::new(0.,  0.0, 0.))).abs() < 1e-10);
        assert!((cone.sdf(&Vector3::new(0., -1.0, 0.)) - 1.0).abs() < 1e-10);
        assert!((cone.sdf(&Vector3::new(0.5, 1.0, 0.)) - 1.0 / (5.0f64).sqrt()).abs() < 1e-10);
        assert!((cone.sdf(&Vector3::new(0., 1.0, 0.5)) - 1.0 / (5.0f64).sqrt()).abs() < 1e-10);
    }
}