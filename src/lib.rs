use numpy::PyArray;
use pyo3::{prelude::*, exceptions::PyTypeError};

mod camera;
mod material;
mod ray;
mod scene;

pub(crate) use camera::*;
pub use material::*;
pub use ray::*;
pub use scene::*;
use slotmap::Key;

impl IntoPy<Py<PyAny>> for ObjectID {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        let keydata = self.0.data().as_ffi();
        keydata.into_py(py)
    }
}

impl<'py> FromPyObject<'py> for ObjectID {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        let keydata: u64 = obj.extract()?;
        Ok(Self::from_keydata(keydata))
    }
}

impl IntoPy<Py<PyAny>> for MaterialID {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        let keydata = self.0.data().as_ffi();
        keydata.into_py(py)
    }
}

impl<'py> FromPyObject<'py> for MaterialID {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        let keydata: u64 = obj.extract()?;
        Ok(Self::from_keydata(keydata))
    }
}

#[pyclass]
#[pyo3(name = "PerspectiveCamera")]
struct PyPerspectiveCamera {
    camera: PerspectiveCamera
}

#[pymethods]
impl PyPerspectiveCamera {
    #[new]
    fn new(params: &PyPerspectiveCameraParams) -> Self {
        Self {
            camera: PerspectiveCamera::new(params.inner.clone()) 
        }
    }

    fn render<'py>(&self, py: Python<'py>, scene: &PyScene) -> &'py numpy::PyArray3<u8> {
        let pixels = self.camera.get_pixels();
        let mut buf = vec![0u32; pixels.0 * pixels.1];
        self.camera.render(&scene.scene, &mut buf);
        let buf: &[u8] = bytemuck::cast_slice(&buf);
        let result = PyArray::from_slice(py, buf);
        result.reshape((pixels.0, pixels.1, 4)).unwrap()
        // let result = buf.into_pyarray(py);
        // result.reshape(pixels).unwrap()
    }
}

#[pyclass]
#[pyo3(name = "PerspectiveCameraParams")]
struct PyPerspectiveCameraParams {
    inner: PerspectiveCameraParams
}

#[pymethods]
impl PyPerspectiveCameraParams {
    #[new]
    fn new() -> Self {
        Self {
            inner: PerspectiveCameraParams::default()
        }
    }

    #[setter]
    fn set_pixels(&mut self, value: (usize, usize)) {
        self.inner.pixels = value
    }

    #[setter]
    fn set_orientation(&mut self, value: (f64, f64, f64)) {
        self.inner.orientation = value.into()
    }
}

#[pyclass]
#[pyo3(name = "Scene")]
struct PyScene {
    scene: Scene
}

#[pymethods]
impl PyScene {
    #[new]
    fn new(sky_color: (f32,f32,f32)) -> Self {
        Self {
            scene: Scene::new(sky_color.into())
        }
    }

    fn add_material(&mut self, py: Python, mat: PyObject) -> Result<MaterialID, PyErr> {
        if let Ok(mat) = mat.downcast::<PyCell<PySimpleMaterial>>(py) {
            Ok(self.scene.add_material(Box::new(mat.borrow().inner.clone())))
        }
        else if let Ok(mat) = mat.downcast::<PyCell<PyCheckerboardMaterial>>(py) {
            Ok(self.scene.add_material(Box::new(mat.borrow().inner.clone())))
        }
        else if let Ok(mat) = mat.downcast::<PyCell<PyFresnelMaterial>>(py) {
            Ok(self.scene.add_material(Box::new(mat.borrow().inner.clone())))
        }
        else if let Ok(mat) = mat.downcast::<PyCell<PyLinearGRINFresnelMaterial>>(py) {
            Ok(self.scene.add_material(Box::new(mat.borrow().inner.clone())))
        }
        else {
            Err(PyTypeError::new_err("invalid argument"))
        }
    }

    fn add_object(&mut self, py: Python, obj: PyObject, mat: MaterialID) -> Result<ObjectID, PyErr> {
        if let Ok(obj) = obj.downcast::<PyCell<PySphere>>(py) {
            Ok(self.scene.add_object(Box::new(obj.borrow().inner.clone()), mat))
        }
        else if let Ok(obj) = obj.downcast::<PyCell<PyPlane>>(py) {
            Ok(self.scene.add_object(Box::new(obj.borrow().inner.clone()), mat))
        }
        else {
            Err(PyTypeError::new_err("invalid argument"))
        }
    }
}

#[pyclass]
#[pyo3(name = "Sphere")]
struct PySphere {
    inner: Sphere
}

#[pymethods]
impl PySphere {
    #[new]
    fn new(center: (f64,f64,f64), radius: f64) -> Self {
        Self {
            inner: Sphere::new(center.into(), radius)
        }
    }
}

#[pyclass]
#[pyo3(name = "Plane")]
struct PyPlane {
    inner: Plane
}

#[pymethods]
impl PyPlane {
    #[new]
    fn new(origin: (f64,f64,f64), normal: (f64,f64,f64)) -> Self {
        Self {
            inner: Plane::new(origin.into(), normal.into())
        }
    }
}

#[pyclass]
#[pyo3(name = "SimpleMaterial")]
struct PySimpleMaterial {
    inner: SimpleMaterial
}

#[pymethods]
impl PySimpleMaterial {
    #[new]
    fn new(color: (f32,f32,f32)) -> Self {
        Self {
            inner: SimpleMaterial::new(color.into())
        }
    }
}

#[pyclass]
#[pyo3(name = "CheckerboardMaterial")]
struct PyCheckerboardMaterial {
    inner: CheckerboardMaterial
}

#[pymethods]
impl PyCheckerboardMaterial {
    #[new]
    fn new(color: (f32,f32,f32), origin: (f32,f32,f32), direction: (f32,f32,f32)) -> Self {
        Self {
            inner: CheckerboardMaterial::new(color.into(), origin.into(), direction.into())
        }
    }
}

#[pyclass]
#[pyo3(name = "FresnelMaterial")]
struct PyFresnelMaterial {
    inner: FresnelMaterial
}

#[pymethods]
impl PyFresnelMaterial {
    #[new]
    fn new(index: f64, outer_index: f64) -> Self {
        Self {
            inner: FresnelMaterial::new(index, outer_index)
        }
    }
}

#[pyclass]
#[pyo3(name = "LinearGRINFresnelMaterial")]
struct PyLinearGRINFresnelMaterial {
    inner: LinearGRINFresnelMaterial
}

#[pymethods]
impl PyLinearGRINFresnelMaterial {
    #[new]
    fn new(reference_index: f64, gradient: (f64, f64, f64), outer_index: f64) -> Self {
        Self {
            inner: LinearGRINFresnelMaterial::new(reference_index, gradient.into(), outer_index)
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn grinray(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPerspectiveCamera>()?;
    m.add_class::<PyPerspectiveCameraParams>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PySphere>()?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PySimpleMaterial>()?;
    m.add_class::<PyCheckerboardMaterial>()?;
    m.add_class::<PyFresnelMaterial>()?;
    m.add_class::<PyLinearGRINFresnelMaterial>()?;
    Ok(())
}