use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use cgmath::{InnerSpace, Vector3, Zero};

use crate::{Ray, scene::Scene, ray::RayGraphicsContext};

const MAX_DEPTH: usize = 10;

fn color_vec3_to_u32(color: &Vector3<f32>) -> u32 {
    // NOTE: WE ASSUME LITTLE ENDIAN HERE
    let mut result = 255;
    result = (result << 8) + ((color.z).clamp(0., 0.999) * 256.) as u32;
    result = (result << 8) + ((color.y).clamp(0., 0.999) * 256.) as u32;
    (result << 8) + ((color.x).clamp(0., 0.999) * 256.) as u32
}

pub(crate) trait Camera {
    fn build_screen_ray(&self, ix: usize, iy: usize) -> Ray;
    fn get_pixels(&self) -> (usize, usize);
    fn render(&self, scene: &Scene, buf: &mut[u32]) {
        let pixels = self.get_pixels();
        let mut ctx = RayGraphicsContext {
            scene,
            rng: Xoshiro256Plus::from_entropy()
        };
        for iy in 0..pixels.1 {
            for ix in 0..pixels.0 {
                let ray = self.build_screen_ray(ix, iy);
                let mut color = Vector3::zero();
                const SAMPLES: usize = 32;
                for _ in 0..SAMPLES {
                    color += ray.get_color(&mut ctx);
                }
                color = color / SAMPLES as f32;
                buf[iy * pixels.0 + ix] = color_vec3_to_u32(&color);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct PerspectiveCameraParams {
    pub(crate) eye: Vector3<f64>,
    pub(crate) orientation: Vector3<f64>,
    pub(crate) up: Vector3<f64>,
    pub(crate) near: f64,
    pub(crate) fov: f64,
    pub(crate) pixels: (usize, usize),
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
        }
    }
}

pub(crate) struct PerspectiveCamera {
    eye: Vector3<f64>,
    up: Vector3<f64>,
    pixels: (usize, usize),
    cell_size: (f64, f64),
    sideways: Vector3<f64>,
    rel_screen_origin: Vector3<f64>,
}

impl PerspectiveCamera {
    pub(crate) fn new(params: PerspectiveCameraParams) -> Self {
        // Validation
        let eye = params.eye;
        let orientation = params.orientation.normalize();
        let up = params.up.normalize();
        assert!(params.near > 0.0);
        let near = params.near;
        assert!(params.fov > 0.0 && params.fov < 180.0);
        let fov = params.fov;
        let pixels = params.pixels;
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
            cell_size,
            sideways,
            rel_screen_origin,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn build_screen_ray(&self, ix: usize, iy: usize) -> Ray {
        let start = self.eye;
        let dir = (self.rel_screen_origin
            + self.cell_size.0 * (0.5 + ix as f64) * self.sideways
            + self.cell_size.1 * (0.5 + iy as f64) * self.up).normalize();
        Ray { start, dir, depth: MAX_DEPTH }
    }

    fn get_pixels(&self) -> (usize, usize) {
        self.pixels
    }
}
