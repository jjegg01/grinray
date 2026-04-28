mod line_segments;
mod mixed_segments;

pub use line_segments::*;
pub use mixed_segments::*;

use cgmath::{InnerSpace, Vector2, Vector3, Zero};

use super::{AABox, SDFObject, SURFACE_EPSILON};

/// Default implementation of the gradient for a 2D radial SDF
pub(crate) fn default_grad_sdf2d_radial<T: RotationallyClosedSDF + ?Sized>(obj: &T, position: &Vector2<f64>) -> Vector2<f64> {
    let h = SURFACE_EPSILON;
    let fyf = obj.sdf(&(position.x, position.y + h / 2.).into());
    let fyb = obj.sdf(&(position.x, position.y - h / 2.).into());
    let dfy = (fyf - fyb) / h;
    // Since negative values for x (i.e., the radial coordinate) are not allowed, we fall back to
    // a simple forward difference if the position is too close to the y-axis
    let dfx = if position.x > h / 2. {
        let fxf = obj.sdf(&(position.x + h / 2., position.y).into());
        let fxb = obj.sdf(&(position.x - h / 2., position.y).into());
        (fxf - fxb) / h
    }
    else {
        let fxf = obj.sdf(&(position.x + h, position.y).into());
        let fxc = obj.sdf(&(position.x, position.y).into());
        (fxf - fxc) / h
    };
    Vector2 { x: dfx, y: dfy }
}

/// Trait for 2D path SDFs that produce a closed volume when rotated about the y-axis
/// 
/// Notable caveats:
/// - Sign of SDF must be negative when point is in the region enclosed by the path and the y-axis
/// - An implementation may assume that x-coordinate of requested point is always non-negative
pub trait RotationallyClosedSDF {
    /// Distance from the path in 2D. Sign must be negative if the point is between the path and
    /// the y-axis.
    fn sdf(&self, position: &Vector2<f64>) -> f64;
    /// Gradient of the SDF in 2D
    fn sdf_grad(&self, position: &Vector2<f64>) -> Vector2<f64> {
        default_grad_sdf2d_radial(self, position)
    }
    /// Bounding box of the 2D SDF given as
    /// `(<max coordinate in x>, (<min coordinate in y>, <max coordinate in y>))`
    fn bounding_box(&self) -> (f64, (f64,f64));
}

impl SDFObject for dyn RotationallyClosedSDF {
    fn sdf(&self, position: &Vector3<f64>) -> f64 {
        // Project 3d position to distance from y-axis and y-coordinate
        let position_2d = Vector2::new((position.x * position.x + position.z * position.z).sqrt(), position.y);
        self.sdf(&position_2d)
    }

    fn sdf_grad(&self, position: &Vector3<f64>) -> Vector3<f64> {
        let position_2d = Vector2::new((position.x * position.x + position.z * position.z).sqrt(), position.y);
        let grad_2d = self.sdf_grad(&position_2d);
        let unit_r = if position_2d.x > 1e-10 {
            Vector3::new(position.x, 0., position.z).normalize()
        }
        else {
            Vector3::zero()
        };
        grad_2d.x * unit_r + grad_2d.y * Vector3::unit_y()
    }

    fn bounding_box(&self) -> super::AABox {
        let (r_max, (y_min, y_max)) = self.bounding_box();
        AABox {
            xlo: -r_max - SURFACE_EPSILON * 1.1,
            xhi: r_max + SURFACE_EPSILON * 1.1,
            ylo: y_min - SURFACE_EPSILON * 1.1,
            yhi: y_max + SURFACE_EPSILON * 1.1,
            zlo: -r_max - SURFACE_EPSILON * 1.1,
            zhi: r_max + SURFACE_EPSILON * 1.1,
        }
    }
}