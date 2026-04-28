use cgmath::{InnerSpace, Matrix2, MetricSpace, SquareMatrix, Vector2};

use super::{distance_from_line_segment, RotationallyClosedSDF};

#[derive(Debug, Clone)]
pub enum Segment {
    Line {
        next_vertex: Vector2<f64>,
    },
    CircleArc {
        next_vertex: Vector2<f64>,
        radius: f64,
    },
}

#[derive(Debug, Clone)]
enum SegmentInternal {
    Line {
        start: Vector2<f64>,
        end: Vector2<f64>,
    },
    CircleArc(CircleArc),
}

#[derive(Debug, Clone)]
struct CircleArc {
    start: Vector2<f64>,
    end: Vector2<f64>,
    center: Vector2<f64>,
    decomp_matrix: Matrix2<f64>,
    radius: f64,
}

impl Segment {
    fn get_next_vertex(&self) -> &Vector2<f64> {
        match self {
            Segment::Line { next_vertex } => next_vertex,
            Segment::CircleArc { next_vertex, .. } => next_vertex,
        }
    }

    fn into_internal(self, current_vertex: Vector2<f64>) -> SegmentInternal {
        match self {
            Segment::Line { next_vertex } => SegmentInternal::Line {
                start: current_vertex,
                end: next_vertex,
            },
            Segment::CircleArc {
                next_vertex,
                radius,
            } => {
                let start = current_vertex;
                let end = next_vertex;

                // -- Calculate the position of the center of the circle (arc) --

                let dist_vec = end - start; // Distance vector start to end
                let dist = start.distance(end); // Distance between start to end
                                                // Orthogonal unit vector to distance vector
                                                // Essentially cross product with unit vector in z, but we flip the sign so this
                                                // vector always points to the *left* of the line between start and end vertex
                let ortho_dist_vec = Vector2::new(-dist_vec.y, dist_vec.x).normalize();
                // The center is calculated by decomposition into a component parallel to the line
                // between start and end vertex and one orthogonal component. The length of these
                // components can be determined via trigonometry (remember: sin(arccos(x))=(1-x^2)^0.5)
                // We also use the sign of the radius to determine the orientation of the arc:
                // Positive sign indicates arcing to the outside direction while a negative sign
                // indicates arcing to the inside
                let center = start + 0.5 * dist_vec
                    + ortho_dist_vec * radius * (1. - (dist / 2. / radius).powi(2)).sqrt();
                let center_to_start = start - center;
                let center_to_end = end - center;
                // -- Calculate decomposition matrix --
                // Definition: Let C be the center of the arc with endpoints A and B.
                // Then decomposition matrix A exist such that for every point P
                // [a;b] = A (P - C) such that (P - C) = a * (A - C) + b * (B - C)
                let inv_decomp_matrix = Matrix2::from_cols(center_to_start, center_to_end);
                let decomp_matrix = inv_decomp_matrix.invert().expect("Invalid circle arc: angle must be between 0 and 180 degrees (non-inclusive!)");


                SegmentInternal::CircleArc(CircleArc {
                    start,
                    end,
                    center,
                    radius,
                    decomp_matrix
                })
            }
        }
    }
}

fn point_above_arc(point: &Vector2<f64>, decomp_matrix: &Matrix2<f64>) -> bool {
    let decomp = decomp_matrix * point;
    decomp.x > 0. && decomp.y > 0.
}

impl SegmentInternal {
    fn calc_bbox(&self) -> ((f64, f64), (f64, f64)) {
        match self {
            SegmentInternal::Line { start, end } => {
                // The bbox of a line is just the rectangle defined by its endpoints
                let x_min = start.x.min(end.x);
                let x_max = start.x.max(end.x);
                let y_min = start.y.min(end.y);
                let y_max = start.y.max(end.y);
                ((x_min, x_max), (y_min, y_max))
            }
            SegmentInternal::CircleArc(CircleArc {
                start,
                end,
                center,
                radius,
                decomp_matrix
            }) => {
                // Initial bounding box defined by start and end vertex
                let mut x_min = start.x.min(end.x);
                let mut x_max = start.x.max(end.x);
                let mut y_min = start.y.min(end.y);
                let mut y_max = start.y.max(end.y);
                // Now check the four points on the circle that touch its bounding box:
                // If they are on the arc segment, we consider them for the bbox
                let circle_left = center - radius.abs() * Vector2::unit_x();
                let circle_right = center + radius.abs() * Vector2::unit_x();
                let circle_bottom = center - radius.abs() * Vector2::unit_y();
                let circle_top = center + radius.abs() * Vector2::unit_y();
                for point in [circle_left, circle_right, circle_bottom, circle_top] {
                    let center_to_point = point - center;
                    // Point is on the arc if its distance vector to the center
                    // can be decomposed into *positive* components of the distance
                    // vectors to the end points
                    if point_above_arc(&center_to_point, decomp_matrix)
                    // center_to_point.dot(*center_to_start) >= 0.
                    //     && center_to_point.dot(*center_to_end) >= 0.
                    {
                        x_min = x_min.min(point.x);
                        x_max = x_max.max(point.x);
                        y_min = y_min.min(point.y);
                        y_max = y_max.max(point.y);
                    }
                }
                ((x_min, x_max), (y_min, y_max))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MixedSegmentsRotationallyClosed {
    segments: Vec<SegmentInternal>,
    bbox: (f64, (f64, f64)),
}

#[derive(Debug)]
pub enum MixedSegmentsRotationallyClosedError {
    /// The line segments do not enclose a bounded volume when rotated about the y-axis
    /// (remember to check the vertex order!)
    Unbounded,
    /// The line segments do not enclose any area and will thus not produce a volume when rotated
    ZeroArea,
}

impl MixedSegmentsRotationallyClosed {
    pub fn new(
        start: f64,
        segments: Vec<Segment>,
    ) -> Result<Self, MixedSegmentsRotationallyClosedError> {
        // Validate that we have at least one segment (rotating a single point does not give a volume)
        if segments.len() < 1 {
            return Err(MixedSegmentsRotationallyClosedError::ZeroArea);
        }
        // Validate that if we have exactly one segment, it must not be a line segment
        if segments.len() == 1 {
            if let Segment::Line { .. } = segments[0] {
                return Err(MixedSegmentsRotationallyClosedError::ZeroArea);
            }
        }
        // Validate that the path ends on the y-axis and the vertices are given anti-clockwise
        let last_vertex = segments.last().unwrap().get_next_vertex();
        if last_vertex.x != 0. {
            return Err(MixedSegmentsRotationallyClosedError::Unbounded);
        }
        if start > last_vertex.y {
            return Err(MixedSegmentsRotationallyClosedError::Unbounded);
        }
        // Build internal segments list with some pre-calculations and determine
        // the bounding box
        let mut segments_internal = Vec::with_capacity(segments.len());
        let mut x_max = 0f64;
        let mut y_min = start;
        let mut y_max = last_vertex.y;
        let mut current_vertex = Vector2::new(0., start);
        for segment in segments {
            let next_vertex = *segment.get_next_vertex();
            let segment = segment.into_internal(current_vertex);
            let ((_, seg_x_max), (seg_y_min, seg_y_max)) = segment.calc_bbox();
            x_max = x_max.max(seg_x_max);
            y_min = y_min.min(seg_y_min);
            y_max = y_max.max(seg_y_max);
            segments_internal.push(segment);
            current_vertex = next_vertex;
        }
        let segments = segments_internal;
        let bbox = (x_max, (y_min, y_max));
        Ok(Self { segments, bbox })
    }
}

fn distance_from_arc_segment(arc: &CircleArc, point: &Vector2<f64>) -> f64 {
    // First, determine if the point is "above" the arc (i.e., within the triangle
    // spanned by the center of the card and the two end points stretched to infinity)
    let center_to_point = point - arc.center;
    if point_above_arc(&center_to_point, &arc.decomp_matrix)
        // center_to_point.dot(arc.center_to_start) >= 0.
        // && center_to_point.dot(arc.center_to_end) >= 0.
    {
        // If the point is above the arc we can take the distance to the center and normalize
        // it to the radius of the arc to get the closest point on the arc
        let closest_point = arc.center + center_to_point.normalize() * arc.radius.abs();
        closest_point.distance(*point)
    } else {
        // Else, the closest point must be either the start or the end point
        // Just calculate both and take the smaller distance
        let dist_start = point.distance(arc.start);
        let dist_end = point.distance(arc.end);
        dist_start.min(dist_end)
    }
}

impl RotationallyClosedSDF for MixedSegmentsRotationallyClosed {
    fn sdf(&self, position: &Vector2<f64>) -> f64 {
        // Basically the same algorithm as for line segments with a minor modification
        let mut d_abs = f64::INFINITY;
        let mut sign = 1.0;
        let mut last_vertex = match self.segments.last().as_ref().unwrap() {
            SegmentInternal::Line { end, .. } => *end,
            SegmentInternal::CircleArc(arc) => arc.end,
        };

        for segment in &self.segments {
            // Calculate distance to segment as well as get start and end point
            let (start, end, d_seg) = match segment {
                SegmentInternal::Line { start, end } => {
                    (start, end, distance_from_line_segment(start, end, position))
                }
                SegmentInternal::CircleArc(arc) => {
                    (&arc.start, &arc.end, distance_from_arc_segment(arc, position))
                },
            };
            // For the sign we use almost the algorithm as with just lines
            // The reason is that even for circular arcs in most cases only the intersection with the line between
            // start and end point matter. Any intersection with the arc that does not produce also
            // an intersection with this line always implies a second intersection with the arc
            // thus negating any sign changes. The only exeption are points IN the the circular arc,
            // for which we need to inverse the sign change.
            if position.y == start.y {
                // Special case: We hit the current vertex exactly.
                if (start.y - last_vertex.y).signum()
                    != (start.y - end.y).signum()
                {
                    if start.x > position.x {
                        sign *= -1.;
                    }
                }
            } else if position.y == end.y {
                // Ignore if end point of current segment is hit (will be dealt with in next iteration)
            } else {
                // Check if the current line segment intersects
                let t = (position.y - start.y) / (end.y - start.y);
                // Is there an intersection?
                if t > 0. && t < 1. {
                    // Is it to the right of the input position?
                    let x = t * end.x + (1. - t) * start.x;
                    // Are we inside the circular arc?
                    let inside_arc = match &segment {
                        SegmentInternal::Line { .. } => false,
                        SegmentInternal::CircleArc(CircleArc { center, start, end, radius, .. }) => {
                            let start_to_end = end - start;
                            let start_to_position = position - start;
                            center.distance(*position) <= radius.abs() && (start_to_end.x * start_to_position.y - start_to_end.y * start_to_position.x) * radius.signum() < 0.
                        },
                    };
                    if x > position.x {
                        // Flip sign
                        sign *= -1.;
                    }
                    if inside_arc {
                        // Flip sign
                        sign *= -1.;
                    }
                }
            }
            // Calculate distance and fold into minimum
            d_abs = d_abs.min(d_seg.abs());
            // Rotate last vertex
            last_vertex = *start;
        }
        // Return minimum distance (with negative sign if we are inside)
        d_abs * sign
    }

    fn bounding_box(&self) -> (f64, (f64, f64)) {
        self.bbox
    }
}
