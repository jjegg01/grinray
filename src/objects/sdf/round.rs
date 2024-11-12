//! Wrapper for an SDF object that applies a rounding to all edges

use super::{AABox, SDFObject};

/// Wrapper for an SDF object that applies a rounding with a given radius to all edges, effectively
/// "smoothing out" the wrapped geometry.
pub struct Rounding<S: SDFObject> {
    inner: S,
    bbox: AABox,
    radius: f64
}

impl<S: SDFObject> Rounding<S> {
    /// Create a new wrapper around a given SDF geometry with a given rounding radius
    /// 
    /// NOTE: Rounding an object this way adds `radius` to the objects size!
    pub fn new(inner: S, radius: f64) -> Self {
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

/// Trait for SDF objects that can be rescaled such that the bounding box of the object is resized
/// by an absolute value uniformly in each dimension
/// 
/// Note: "Uniform" refers to the fact that each dimension of the bounding box is changed by the
/// same amount. It does *not* mean a uniform/isotropic scaling, i.e., the aspect ratio of the
/// bounding box may change.
pub trait RescaleAbsoluteUniform {
    /// Scale the object such that its bounding box increases or decreases by the given value in
    /// every dimension
    fn rescale_absolute(&mut self, size_change: f64);
}

impl<S: SDFObject + RescaleAbsoluteUniform> Rounding<S> {
    /// Create a new wrapper around a given SDF geometry with a given rounding radius
    pub fn new_autoscale(mut inner: S, radius: f64) -> Self {
        let bbox = inner.bounding_box();
        inner.rescale_absolute(-2. * radius);
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