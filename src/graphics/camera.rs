use cgmath::{InnerSpace, Vector3, Zero};

use crate::{graphics::{util, RayGraphicsContext}, Ray, Tracer};

// -- Generic camera trait --

pub trait Camera<T: Tracer> {
    /// Build the ray corresponding to the given pixel coordinates
    fn build_screen_ray(&self, ix: usize, iy: usize) -> Ray;
    /// Get size of the image in pixels
    fn get_image_size(&self) -> (usize, usize);
    /// Get number of samples for each pixel
    fn get_samples(&self) -> usize;
    /// Render a given scene with this camera into a given image buffer
    fn render<'a>(&self, ctx: &mut RayGraphicsContext<'a, T>, buf: &mut [u32], tracer: &mut T) {
        let pixels = self.get_image_size();
        let num_samples = self.get_samples();
        for iy in 0..pixels.1 {
            for ix in 0..pixels.0 {
                // Debugging helper
                // if ix == pixels.0 / 20 * 5 && iy == pixels.1 / 10 * 4 {
                //     // Color one pixel red to calibrate the condition
                //     buf[iy * pixels.0 + ix] = util::color_vec3_to_u32(&Vector3::new(1.0, 0., 0.));
                //     continue;
                //     // Debug message to break on
                //     dbg!("!");
                // }
                let ray = self.build_screen_ray(ix, iy);
                let mut color = Vector3::zero();
                for _ in 0..num_samples {
                    let trace = tracer.new_trace(ray.start);
                    color += ray.get_color(ctx, tracer, trace);
                }
                color = color / num_samples as f32;
                buf[iy * pixels.0 + ix] = util::color_vec3_to_u32(&color);
            }
        }
    }
}

// -- Orthgraphic camera implementation --

pub struct OrthographicCamaraParameters {
    /// Position of the camera "eye", i.e., the center of the projection plane
    pub eye: Vector3<f64>,
    /// Orientation of the camera, i.e., the view direction
    pub orientation: Vector3<f64>,
    /// Orientation of the up vector of the camera to define the y-axis of the resulting image
    pub up: Vector3<f64>,
    /// Total extent of the projection plane in the y-direction (extent in x in inferred from aspect ratio)
    pub projection_height: f64,
    /// The number of pixels in the resulting image
    pub pixels: (usize, usize),
    /// The number of samples to take for each ray
    pub samples: usize,
    /// The maximum depth (i.e., number of interactions) allowed for each ray before it is discarded
    pub max_depth: usize
}

impl Default for OrthographicCamaraParameters {
    fn default() -> Self {
        Self {
            eye: Vector3::zero(),
            orientation: -Vector3::unit_z(),
            up: Vector3::unit_y(),
            projection_height: 4.0,
            pixels: (256, 256),
            samples: 32,
            max_depth: 10
        }
    }
}

pub struct OrthographicCamera {
    orientation: Vector3<f64>,
    up: Vector3<f64>,
    sideways: Vector3<f64>,
    cell_size: (f64, f64),
    screen_origin: Vector3<f64>,
    pixels: (usize, usize),
    samples: usize,
    max_depth: usize
}

impl OrthographicCamera {
    pub fn new(parameters: OrthographicCamaraParameters) -> Self {
        // Make sure all the directional vectors are normalized
        let orientation = parameters.orientation.normalize();
        let up = parameters.up.normalize();
        // Determine the equivalent of the up vector for the x-direction
        let sideways = orientation.cross(up);
        // Determine screensize from aspect ratio
        let pixels = parameters.pixels;
        let screen_size_y = parameters.projection_height;
        let screen_size_x = screen_size_y * pixels.0 as f64 / pixels.1 as f64;
        // Determine the size of each pixel cell on the projection plane
        let cell_size = (
            screen_size_x / pixels.0 as f64,
            screen_size_y / pixels.1 as f64,
        );
        // Calculate lower left corner of the projection screen
        let screen_origin = parameters.eye - screen_size_x / 2.0 * sideways - screen_size_y / 2.0 * up;
        Self {
            orientation,
            up,
            sideways,
            cell_size,
            screen_origin,
            pixels,
            samples: parameters.samples,
            max_depth: parameters.max_depth
        }
    }
}

impl<T: Tracer> Camera<T> for OrthographicCamera {
    fn build_screen_ray(&self, ix: usize, iy: usize) -> Ray {
        // The ray starts in the projection plane at the center of the pixel cell
        let start = self.screen_origin
            + self.cell_size.0 * (0.5 + ix as f64) * self.sideways
            + self.cell_size.1 * (0.5 + iy as f64) * self.up;
        Ray { start, dir: self.orientation, depth: self.max_depth }
    }

    fn get_image_size(&self) -> (usize, usize) {
        self.pixels
    }

    fn get_samples(&self) -> usize {
        self.samples
    }
}

// -- Perspective camera implementation --

#[derive(Clone)]
pub struct PerspectiveCameraParameters {
    /// Position of the camera "eye", i.e., the origin of all camera rays
    pub eye: Vector3<f64>,
    /// Orientation of the camera, i.e., the view direction
    pub orientation: Vector3<f64>,
    /// Orientation of the up vector of the camera to define the y-axis of the resulting image
    pub up: Vector3<f64>,
    /// The distance of the near clipping plane (i.e., the "projection screen") from the eye position
    pub near: f64,
    /// The field of view angle, i.e. half of the camera's frustrum angle in the y-direction
    pub fov: f64,
    /// The number of pixels in the resulting image
    pub pixels: (usize, usize),
    /// The number of samples to take for each ray
    pub samples: usize,
    /// The maximum depth (i.e., number of interactions) allowed for each ray before it is discarded
    pub max_depth: usize
}

impl Default for PerspectiveCameraParameters {
    fn default() -> Self {
        Self {
            eye: Vector3::zero(),
            orientation: -Vector3::unit_z(),
            up: Vector3::unit_y(),
            near: 1.0,
            fov: 50.0,
            pixels: (256, 256),
            samples: 32,
            max_depth: 10
        }
    }
}

pub struct PerspectiveCamera {
    eye: Vector3<f64>,
    up: Vector3<f64>,
    pixels: (usize, usize),
    samples: usize,
    cell_size: (f64, f64),
    sideways: Vector3<f64>,
    rel_screen_origin: Vector3<f64>,
    max_depth: usize
}

impl PerspectiveCamera {
    pub fn new(parameters: PerspectiveCameraParameters) -> Self {
        // -- Validation --
        let eye = parameters.eye;
        let orientation = parameters.orientation.normalize();
        let up = parameters.up.normalize();
        assert!(parameters.near > 0.0);
        let near = parameters.near;
        assert!(parameters.fov > 0.0 && parameters.fov < 180.0);
        let fov = parameters.fov;
        let pixels = parameters.pixels;
        let samples = parameters.samples;
        let max_depth = parameters.max_depth;
        // -- Actual calculation --
        // First determine the screen size in world coordinates
        let screen_size_y = 2.0 * near * fov.to_radians().tan();
        // The size in x is determined by the output image aspect ratio
        let screen_size_x = screen_size_y * pixels.0 as f64 / pixels.1 as f64;
        // The size of each pixel cell on the projection plane
        let cell_size = (
            screen_size_x / pixels.0 as f64,
            screen_size_y / pixels.1 as f64,
        );
        // Determine the equivalent of the up vector for the x-direction
        let sideways = orientation.cross(up);
        // Calculate the vector from the eye position to the lower left corner of the projection screen
        let rel_screen_origin =
            near * orientation - screen_size_x / 2.0 * sideways - screen_size_y / 2.0 * up;
        Self {
            eye,
            up,
            pixels,
            samples,
            cell_size,
            sideways,
            rel_screen_origin,
            max_depth
        }
    }
}

impl<T: Tracer> Camera<T> for PerspectiveCamera {
    fn build_screen_ray(&self, ix: usize, iy: usize) -> Ray {
        let start = self.eye;
        let dir = (self.rel_screen_origin
            + self.cell_size.0 * (0.5 + ix as f64) * self.sideways
            + self.cell_size.1 * (0.5 + iy as f64) * self.up)
            .normalize();
        Ray {
            start,
            dir,
            depth: self.max_depth,
        }
    }

    fn get_image_size(&self) -> (usize, usize) {
        self.pixels
    }

    fn get_samples(&self) -> usize {
        self.samples
    }
}
