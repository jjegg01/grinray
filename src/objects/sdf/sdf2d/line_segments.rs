use core::f64;

use cgmath::{InnerSpace, MetricSpace, Vector2, Zero};

use super::RotationallyClosedSDF;

/// A series of line segments that produce a closed shape when rotated about the y-axis.
/// 
/// Notable caveats:
/// - Points must be given anti-clockwise, i.e., the start point must be at a lower y coordinate
///   than then end point
/// - No self-intersection allowed
#[derive(Clone)]
pub struct LineSegmentsRotationallyClosed {
    /// y-coordinate of start point
    start: f64,
    /// coordinates of intermediate points
    intermediates: Vec<Vector2<f64>>,
    /// y-coordinate of end point
    end: f64,
    /// Bounding box
    bbox: (f64, (f64,f64))
}

#[derive(Debug)]
pub enum LineSegmentsRotationallyClosedError {
    /// The line segments do not enclose a bounded volume when rotated about the y-axis
    Unbounded,
    /// The line segments do not enclose any area and will thus not produce a volume when rotated
    ZeroArea
}

impl LineSegmentsRotationallyClosed {
    /// Create a new sequence of line segments. Since the start and the end point must be on the
    /// y-axis, only the x-coordinate is required to initialize them. There must be at least one
    /// intermediate point.
    /// 
    /// Note: negative x-coordinates for the intermediates are automatically flipped in sign as
    /// negative values for the radial coordinate make little sense.
    pub fn new(start: f64, mut intermediates: Vec<Vector2<f64>>, end: f64) -> Result<Self, LineSegmentsRotationallyClosedError> {
        // Validate that line is defined clockwise
        if start > end {
            return Err(LineSegmentsRotationallyClosedError::Unbounded)
        }
        // Validate that we have at least one intermediate
        if intermediates.len() < 1 {
            return Err(LineSegmentsRotationallyClosedError::ZeroArea)
        }
        // Check signs of x coordinates and find bounding box
        let mut x_max = 0f64;
        let mut y_min = start;
        let mut y_max = end;
        for intermediate in &mut intermediates {
            if intermediate.x < 0. {
                intermediate.x *= -1.;
            }
            x_max = x_max.max(intermediate.x);
            y_min = y_min.min(intermediate.y);
            y_max = y_max.max(intermediate.y);
        }
        let bbox = (x_max, (y_min, y_max));
        Ok(Self { start, intermediates, end, bbox })
    }
}

/// Helper function to calculate the distance of point C from the line segment enclosed by points
/// A and B
fn distance_from_line_segment(a: &Vector2<f64>, b: &Vector2<f64>, c: &Vector2<f64>) -> f64 {
    let ac = c - a;
    let ab = b - a;
    let t = (ac.dot(ab) / ab.distance2(Vector2::zero())).clamp(0., 1.);
    (ab * t - ac).distance(Vector2::zero())
}

impl RotationallyClosedSDF for LineSegmentsRotationallyClosed {
    fn sdf(&self, position: &Vector2<f64>) -> f64 {
        let start = Vector2::new(0., self.start);
        let end = Vector2::new(0., self.end);
        let c = position;
        let mut d_abs = f64::INFINITY;
        let mut inside = true;
        // Iterate over all line segments
        for i in 0..(self.intermediates.len() + 1) {
            // Select segment
            let a = if i == 0 { &start } else { &self.intermediates[i-1] };
            let b = if i == self.intermediates.len() { &end } else { &self.intermediates[i] };
            // Calculate distance and fold into minimum
            let d_seg = distance_from_line_segment(&a, &b, c);
            d_abs = d_abs.min(d_seg.abs());
            // Check if point is to the left of the current segment
            let ac = c - a;
            let ab = b - a;
            let cross = ab.x * ac.y - ab.y * ac.x;
            if cross < 0. {
                inside = false;
            }
        }
        // Return minimum distance (with negative sign if we are inside)
        if inside {
            d_abs * -1.
        }
        else {
            d_abs
        }
    }

    fn bounding_box(&self) -> (f64, (f64,f64)) {
        self.bbox.clone()
    }
}