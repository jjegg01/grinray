//! Traits and implementations for debugging or visualizing the raytracing process

use std::hash::Hash;

use cgmath::Vector3;
use slotmap::{new_key_type, SlotMap};

/// Different kinds of events that can occur along the path of a light ray
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum TraceEvent {
    /// This indicates the start of the ray
    Start,
    /// This indicates some intermediate point during ray travel
    /// Note: This event should only be generated if the ray is not traveling in a straight line
    Intermediate,
    /// The ray is reflected or scattered off of a surface
    Reflection,
    /// The ray is entering another medium with a different refractive index
    Refraction,
    /// The ray ends (e.g., total absorption, hitting a black hole, etc.)
    /// Note: Typically, rays "end" by exiting the scene and travelling to infinity. This event
    /// is only meant to capture rays that actually have a finite travel distance
    End,
}

#[derive(Debug, Clone)]
pub struct TracePoint {
    pub location: Vector3<f64>,
    pub event: TraceEvent,
}

impl PartialEq for TracePoint {
    fn eq(&self, other: &Self) -> bool {
        // TracePoints should not be NaN, but just in case we use total_cmp to test for equality
        self.location.x.total_cmp(&other.location.x).is_eq()
            && self.location.y.total_cmp(&other.location.y).is_eq()
            && self.location.z.total_cmp(&other.location.z).is_eq()
            && self.event == other.event
    }
}

impl Eq for TracePoint {}

/// A tracer is some kind of container that can store traces, i.e., a series of
/// ray tracing "events" that occur at specific points along the path taken by
/// a light ray.
pub trait Tracer {
    type TraceID: Copy;

    /// Create a new, empty tracer
    fn new() -> Self;

    /// Start a new trace from a given point
    fn new_trace(&mut self, start: Vector3<f64>) -> Self::TraceID;

    /// Add a point to given trace
    fn add_point(&mut self, trace: Self::TraceID, event: TraceEvent, location: Vector3<f64>);

    /// Indicate that the given trace has ended to free resources
    /// The behavior when adding a point to a trace that has been ended and ending a trace multiple
    /// times depends on the implementor (may panic!)
    /// This function also indicates that the given trace ID may be reused for new traces
    fn end_trace(&mut self, trace: Self::TraceID);
}

/// Using the unit type as a tracer will just eat all tracing information
impl Tracer for () {
    type TraceID = ();

    fn new() -> Self {
        ()
    }

    fn new_trace(&mut self, _: Vector3<f64>) -> Self::TraceID {
        ()
    }

    fn add_point(&mut self, _trace: Self::TraceID, _event: TraceEvent, _location: Vector3<f64>) {}

    fn end_trace(&mut self, _trace: Self::TraceID) {}
}

/// This tracer will collect all intermediate points and is intended for
/// debugging purposes
pub struct FullTracer {
    traces: Vec<Vec<TracePoint>>,
}

#[derive(Clone, Copy)]
pub struct FullTraceID(usize);

impl Tracer for FullTracer {
    type TraceID = FullTraceID;

    fn new() -> Self {
        Self { traces: vec![] }
    }

    fn new_trace(&mut self, start: Vector3<f64>) -> FullTraceID {
        let trace = vec![TracePoint {
            location: start,
            event: TraceEvent::Start,
        }];
        self.traces.push(trace);
        FullTraceID(self.traces.len() - 1)
    }

    fn add_point(&mut self, trace: Self::TraceID, event: TraceEvent, location: Vector3<f64>) {
        self.traces[trace.0].push(TracePoint { location, event });
    }

    /// No-op implementation: you may still add points to an ended trace
    fn end_trace(&mut self, _trace: Self::TraceID) {}
}

impl FullTracer {
    pub fn get_traces(self) -> Vec<Vec<TracePoint>> {
        self.traces
    }
}

new_key_type! { pub struct CountingTraceID; }

/// A tracer implementation that is optimized for tracing the same path repeatedly (e.g. for
/// probabilistic ray tracing)
pub struct CountingTracer {
    /// Traces that have not been ended yet
    open_traces: SlotMap<CountingTraceID, Vec<TracePoint>>,
    /// Once a trace is ended, it is stored efficiently in a forest structure
    closed_traces: Vec<CountingTreeNode>,
}

#[derive(Debug)]
pub struct CountingTreeNode {
    pub tracepoint: TracePoint,
    pub count: usize,
    pub children: Vec<CountingTreeNode>,
}

impl CountingTreeNode {
    fn new(tracepoint: TracePoint) -> Self {
        Self {
            tracepoint,
            count: 0,
            children: vec![],
        }
    }
}

impl Tracer for CountingTracer {
    type TraceID = CountingTraceID;

    fn new() -> Self {
        Self {
            open_traces: SlotMap::with_key(),
            closed_traces: vec![],
        }
    }

    fn new_trace(&mut self, start: Vector3<f64>) -> Self::TraceID {
        let trace = vec![TracePoint {
            location: start,
            event: TraceEvent::Start,
        }];
        self.open_traces.insert(trace)
    }

    fn add_point(&mut self, trace: Self::TraceID, event: TraceEvent, location: Vector3<f64>) {
        self.open_traces[trace].push(TracePoint { location, event });
    }

    // This function adds the trace to the internal trace tree. If a trace is not terminated via
    // this function, it will not be part of the tree returned by `get_trace_trees`!
    fn end_trace(&mut self, trace: Self::TraceID) {
        // Panic if trace is not open anymore (these tracers are for debugging anyways)
        let trace = self.open_traces.remove(trace).unwrap();
        let mut next_nodes = &mut self.closed_traces;
        for tracepoint in trace {
            // Get next node in tree that fits the current trace point
            let next_node = match next_nodes
                .iter()
                .position(|node| node.tracepoint.eq(&tracepoint))
            {
                Some(idx) => &mut next_nodes[idx],
                None => {
                    // Add current trace point to tree of not already present
                    next_nodes.push(CountingTreeNode::new(tracepoint));
                    let last_idx = next_nodes.len() - 1;
                    &mut next_nodes[last_idx]
                }
            };
            // Increment counter for trace point and recurse deeper
            next_node.count += 1;
            next_nodes = &mut next_node.children;
        }
    }
}

impl CountingTracer {
    pub fn get_trace_trees(self) -> Vec<CountingTreeNode> {
        self.closed_traces
    }
}

/// A special tracer that does nothing by default and need to be triggered by a debugger first
/// 
/// This tracer is intended for situations where the number of rays is too large for the other
/// tracers to handle. At the start of the program, this tracer implementation does nothing as the
/// internal `toggle` field is set to `false`. By flipping this toggle to `true` through a debugger,
/// the tracer starts to output all tracepoints to stderr. Through the use of appropriate break- or
/// watchpoints, this can be used as a very customizable filter for tracing data.
pub struct DebuggerTracer {
    toggle: bool
}

impl Tracer for DebuggerTracer {
    type TraceID = ();

    fn new() -> Self {
        Self { toggle: std::hint::black_box(false) }
    }

    fn new_trace(&mut self, _: Vector3<f64>) -> Self::TraceID {
        ()
    }

    fn add_point(&mut self, _: Self::TraceID, event: TraceEvent, location: Vector3<f64>) {
        if std::hint::black_box(self.toggle) {
            eprintln!("{:?}@[{},{},{}]", event, location.x, location.y, location.z);
        }
    }

    fn end_trace(&mut self, _: Self::TraceID) {
        ()
    }
}