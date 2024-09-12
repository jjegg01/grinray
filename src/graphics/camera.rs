use cgmath::{InnerSpace, Vector3, Zero};

use crate::{graphics::{util, RayGraphicsContext}, Ray, Tracer};

const MAX_DEPTH: usize = 10;

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

#[derive(Clone)]
pub struct PerspectiveCameraParams {
    pub eye: Vector3<f64>,
    pub orientation: Vector3<f64>,
    pub up: Vector3<f64>,
    pub near: f64,
    pub fov: f64,
    pub pixels: (usize, usize),
    pub samples: usize
}

impl Default for PerspectiveCameraParams {
    fn default() -> Self {
        Self {
            eye: Vector3::zero(),
            orientation: -Vector3::unit_z(),
            up: Vector3::unit_y(),
            near: 1.0,
            fov: 50.0,
            pixels: (256, 256),
            samples: 32
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
}

impl PerspectiveCamera {
    pub fn new(params: PerspectiveCameraParams) -> Self {
        // Validation
        let eye = params.eye;
        let orientation = params.orientation.normalize();
        let up = params.up.normalize();
        assert!(params.near > 0.0);
        let near = params.near;
        assert!(params.fov > 0.0 && params.fov < 180.0);
        let fov = params.fov;
        let pixels = params.pixels;
        let samples = params.samples;
        // Actual calculation
        let screen_size_y = 2.0 * near * fov.to_radians().sin();
        let screen_size_x = screen_size_y * pixels.0 as f64 / pixels.1 as f64;
        // let screen_size = (screen_size_x, screen_size_y);
        let cell_size = (
            screen_size_x / pixels.0 as f64,
            screen_size_y / pixels.1 as f64,
        );
        let sideways = orientation.cross(up);
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
            depth: MAX_DEPTH,
        }
    }

    fn get_image_size(&self) -> (usize, usize) {
        self.pixels
    }

    fn get_samples(&self) -> usize {
        self.samples
    }
}
