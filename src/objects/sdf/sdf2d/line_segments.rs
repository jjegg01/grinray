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
    /// Vertices that make up the line segment
    vertices: Vec<Vector2<f64>>,
    /// Bounding box
    bbox: (f64, (f64,f64))
}

#[derive(Debug)]
pub enum LineSegmentsRotationallyClosedError {
    /// The line segments do not enclose a bounded volume when rotated about the y-axis
    /// (remember to check the vertex order!)
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
    pub fn new(start: f64, intermediates: Vec<Vector2<f64>>, end: f64) -> Result<Self, LineSegmentsRotationallyClosedError> {
        // Validate that line is defined anti-clockwise
        if start > end {
            return Err(LineSegmentsRotationallyClosedError::Unbounded)
        }
        // Validate that we have at least one intermediate
        if intermediates.len() < 1 {
            return Err(LineSegmentsRotationallyClosedError::ZeroArea)
        }
        // Check signs of x coordinates and find bounding box
        let mut vertices = Vec::with_capacity(intermediates.len() + 2);
        let mut x_max = 0f64;
        let mut y_min = start;
        let mut y_max = end;
        vertices.push(Vector2::new(0., start));
        for intermediate in &intermediates {
            vertices.push(Vector2::new(intermediate.x.abs(), intermediate.y));
            x_max = x_max.max(intermediate.x);
            y_min = y_min.min(intermediate.y);
            y_max = y_max.max(intermediate.y);
        }
        vertices.push(Vector2::new(0., end));
        let bbox = (x_max, (y_min, y_max));
        Ok(Self { vertices, bbox })
    }
}

/// Helper function to calculate the distance of point C from the line segment enclosed by points
/// A and B
fn distance_from_line_segment(a: &Vector2<f64>, b: &Vector2<f64>, c: &Vector2<f64>) -> f64 {
    let ac = c - a;
    let ab = b - a;
    let t = (ac.dot(ab) / ab.dot(ab)).clamp(0., 1.);
    (ab * t - ac).distance(Vector2::zero())
}

impl RotationallyClosedSDF for LineSegmentsRotationallyClosed {
    fn sdf(&self, position: &Vector2<f64>) -> f64 {
        let mut d_abs = f64::INFINITY;
        let mut sign = 1.0;
        let mut last_vertex = self.vertices[self.vertices.len() - 1];
        // Iterate over all vertices except the last
        // Why not the last one? Because our path is not closed, since the section on the y-axis
        // is not part of the distance calculation. The last vertex is also irrelevant to the sign
        // of the distance as it lies on the y-axis and thus cannot be to the right of the position
        // (strictly speaking, the position could still coincide with the last vertex, but that
        // implies a distance of zero, so the sign is irrelevant)
        for i in 0..(self.vertices.len() - 1) {
            // Each iteration considers the edge between current_vertex and next_vertex
            let current_vertex = self.vertices[i];
            let next_vertex = self.vertices[i+1];
            // To determine the sign of the distance function, we cast a ray to the right and count
            // the number of edges we intersect. If the number is odd, we are inside (negative sign)
            // First check: do we hit the endpoints exactly?
            if position.y == current_vertex.y {
                // Special case: We hit the current vertex exactly. This can only be resolved by looking at
                // the next vertex and comparing to the last. We only count the intersection, if the
                // last vertex and the next vertex are on opposite sides of the ray.
                if (current_vertex.y - last_vertex.y).signum() != (current_vertex.y - next_vertex.y).signum() {
                    if current_vertex.x > position.x {
                        sign *= -1.;
                    }
                }
            }
            else if position.y == next_vertex.y {
                // Ignore if end point of current segment is hit (will be dealt with in next iteration)
            }
            else {
                // Check if the current line segment intersects
                let t = (position.y - current_vertex.y) / (next_vertex.y - current_vertex.y);
                // Is there an intersection?
                if t > 0. && t < 1. {
                    // Is it to the right of the input position?
                    let x = t * next_vertex.x + (1.-t) * current_vertex.x;
                    if x > position.x {
                        // Flip sign
                        sign *= -1.;
                    }
                }
            }
            // Calculate distance and fold into minimum
            let d_seg = distance_from_line_segment(&current_vertex, &next_vertex, position);
            d_abs = d_abs.min(d_seg.abs());
            // Rotate last vertex
            last_vertex = current_vertex
        }
        // Return minimum distance (with negative sign if we are inside)
        d_abs * sign
    }

    fn bounding_box(&self) -> (f64, (f64,f64)) {
        self.bbox.clone()
    }
}