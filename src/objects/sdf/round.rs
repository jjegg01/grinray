//! Wrapper for an SDF object that applies a rounding to all edges

use super::{AABox, SDFObject};

/// Generic trait for SDF-based geometries that support "rounding", i.e., the replacement of sharp
/// edges with curved geometries. The strength of rounding is parametrized by a positive "rounding
/// radius". The exact geometric interpretation of this rounding radius depends on the particular
/// object type and its rounding implementation.
pub trait Roundable {
    type Rounded: SDFObject;

    fn round(self, radius: f64) -> Self::Rounded;
}

/// Wrapper for an SDF object that applies a rounding with a given radius *r* to all edges.
/// This is implemented by subtracting a constant from the SDF of the wrapped object, i.e. the
/// volume of the wrapped object is extended to also include points with a distance of at most *r*
/// from its surface (i.e., the Minkowski sum with a sphere of radius *r*).
/// Effectively, this smoothes out the wrapped geometry as sharp edges are "blown out" into circular
/// arcs. Naturally, this approach increases the size of the wrapped object by *r* in all directions.
pub struct Rounding<S: SDFObject> {
    inner: S,
    bbox: AABox,
    radius: f64
}

impl<S: SDFObject> Rounding<S> {
    /// Create a new wrapper around a given SDF geometry with a given rounding radius
    /// 
    /// NOTE: Rounding an object this way adds `radius` to the objects size!
    pub(crate) fn new(inner: S, radius: f64) -> Self {
        // By subtracting a constant from the SDF we do not just smooth the object but also
        // increase its size => bounding box needs correction
        let original_bbox = inner.bounding_box();
        let bbox = AABox {
            xlo: original_bbox.xlo - radius,
            xhi: original_bbox.xhi + radius,
            ylo: original_bbox.ylo - radius,
            yhi: original_bbox.yhi + radius,
            zlo: original_bbox.zlo - radius,
            zhi: original_bbox.zhi + radius,
        };
        Self { inner, bbox, radius }
    }
}

impl <S: SDFObject> SDFObject for Rounding<S> {
    fn sdf(&self, position: &cgmath::Vector3<f64>) -> f64 {
        self.inner.sdf(position) - self.radius
    }

    fn bounding_box(&self) -> super::AABox {
        self.bbox.clone()
    }
}

/// Trait for SDF objects that can be rescaled such that the bounding box of the object is resized
/// by an absolute value uniformly in each dimension.
/// 
/// Note: "Uniform" refers to the fact that each dimension of the bounding box is changed by the
/// same amount. It does **not** mean a uniform/isotropic scaling, i.e., the aspect ratio of the
/// bounding box may change.
pub trait RescaleAbsoluteUniform {
    /// Scale the object such that its bounding box increases or decreases by the given value in
    /// every dimension
    fn rescale_absolute(&mut self, size_change: f64);
}

impl<S: RescaleAbsoluteUniform + SDFObject> Roundable for S {
    type Rounded = Rounding<Self>;

    /// Blanket implementation for rounding for SDFs that support absolute bbox rescaling.
    fn round(mut self, radius: f64) -> Self::Rounded {
        self.rescale_absolute(-2. * radius);
        Rounding::new(self, radius)
    }
}

#[cfg(test)]
mod test {
    use cgmath::Vector3;

    use crate::objects::sdf::{Cuboid, Roundable, SDFObject};

    #[test]
    fn test_rounding_cube() {
        let rounding_radius = 0.25;
        let cube_length = 2.0;
        let cube = Cuboid::new(cube_length, cube_length, cube_length);
        let rounded_cube = Cuboid::new(cube_length, cube_length, cube_length)
            .round(rounding_radius); // TODO: Make cuboid Clone
        // Midpoint of faces of rounded cube should still have distance of zero
        assert!((rounded_cube.sdf(&Vector3::new(1.0, 0.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(-1.0, 0.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(0.0, 1.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(0.0, -1.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(0.0, 0.0, 1.0)) - 0.0).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(0.0, 0.0, -1.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(1.0, 0.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(-1.0, 0.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(0.0, 1.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(0.0, -1.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(0.0, 0.0, 1.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(0.0, 0.0, -1.0)) - 0.0).abs() < 1e-10);
        // Corners and edges of cube should have a distance larger than zero for rounded cube
        assert!((rounded_cube.sdf(&Vector3::new(1.0, 1.0, -1.0)) - rounding_radius * (3.0f64.sqrt() - 1.)).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(1.0, 1.0, 0.0)) - rounding_radius * (2.0f64.sqrt() - 1.)).abs() < 1e-10);
        assert!((rounded_cube.sdf(&Vector3::new(1.0, 1.0, 1.0)) - rounding_radius * (3.0f64.sqrt() - 1.)).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(1.0, 1.0, -1.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(1.0, 1.0, 0.0)) - 0.0).abs() < 1e-10);
        assert!((cube.sdf(&Vector3::new(1.0, 1.0, 1.0)) - 0.0).abs() < 1e-10);
    }
}
