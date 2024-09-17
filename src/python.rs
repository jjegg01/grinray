use crate::*;
use cgmath::{Quaternion, Rad, Rotation3, Vector3};
use graphics::{Camera, PerspectiveCamera, PerspectiveCameraParameters, RayGraphicsContext};
use objects::{Cuboid, ObjectTransform, Plane, Sphere};
use slotmap::Key;
use std::mem;

use pyo3::{exceptions::PyTypeError, prelude::*, types::PyBytes};

use slotmap::KeyData;
impl ObjectID {
    pub(crate) fn from_keydata(data: u64) -> Self {
        Self(ObjectKey::from(KeyData::from_ffi(data)))
    }
}

impl MaterialID {
    pub(crate) fn from_keydata(data: u64) -> Self {
        Self(MaterialKey::from(KeyData::from_ffi(data)))
    }
}

impl IntoPy<Py<PyAny>> for ObjectID {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        let keydata = self.0.data().as_ffi();
        keydata.into_py(py)
    }
}

impl<'py> FromPyObject<'py> for ObjectID {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
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
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let keydata: u64 = obj.extract()?;
        Ok(Self::from_keydata(keydata))
    }
}

#[pyclass]
#[pyo3(name = "PerspectiveCamera")]
struct PyPerspectiveCamera {
    camera: PerspectiveCamera,
}

#[pymethods]
impl PyPerspectiveCamera {
    #[new]
    fn new(parameters: &PyPerspectiveCameraParameters) -> Self {
        Self {
            camera: PerspectiveCamera::new(parameters.inner.clone()),
        }
    }

    fn render<'py>(&self, py: Python<'py>, scene: &PyScene) -> Bound<'py, PyBytes> {
        let pixels = Camera::<()>::get_image_size(&self.camera);
        let mut buf = vec![0u8; mem::size_of::<u32>() * pixels.0 * pixels.1];
        let mut ctx = RayGraphicsContext::with_defaults(&scene.scene);
        self.camera
            .render(&mut ctx, bytemuck::cast_slice_mut(buf.as_mut_slice()), &mut ());
        PyBytes::new_bound(py, &buf)
    }
}

#[pyclass]
#[pyo3(name = "PerspectiveCameraParameters")]
struct PyPerspectiveCameraParameters {
    inner: PerspectiveCameraParameters,
}

#[pymethods]
impl PyPerspectiveCameraParameters {
    #[new]
    fn new() -> Self {
        Self {
            inner: PerspectiveCameraParameters::default(),
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
    scene: Scene<()>,
}

#[pymethods]
impl PyScene {
    #[new]
    fn new() -> Self {
        Self {
            scene: Scene::new(),
        }
    }

    fn add_material(&mut self, py: Python, mat: PyObject) -> Result<MaterialID, PyErr> {
        if let Ok(mat) = mat.downcast_bound::<PySimpleMaterial>(py) {
            Ok(self
                .scene
                .add_material(Box::new(mat.borrow().inner.clone())))
        } else if let Ok(mat) = mat.downcast_bound::<PyCheckerboardMaterial>(py) {
            Ok(self
                .scene
                .add_material(Box::new(mat.borrow().inner.clone())))
        } else if let Ok(mat) = mat.downcast_bound::<PyFresnelMaterial>(py) {
            Ok(self
                .scene
                .add_material(Box::new(mat.borrow().inner.clone())))
        } else if let Ok(mat) = mat.downcast_bound::<PyLinearGRINFresnelMaterial>(py) {
            Ok(self
                .scene
                .add_material(Box::new(mat.borrow().inner.clone())))
        } else {
            Err(PyTypeError::new_err("invalid argument"))
        }
    }

    fn add_object<'py>(
        &mut self,
        py: Python,
        obj: PyObject,
        transform: &'py PyObjectTransform,
        mat: MaterialID,
    ) -> Result<ObjectID, PyErr> {
        if let Ok(obj) = obj.downcast_bound::<PySphere>(py) {
            Ok(self
                .scene
                .add_object(Box::new(obj.borrow().inner.clone()), transform.inner.clone(), mat))
        } else if let Ok(obj) = obj.downcast_bound::<PyPlane>(py) {
            Ok(self
                .scene
                .add_object(Box::new(obj.borrow().inner.clone()), transform.inner.clone(), mat))
        } else if let Ok(obj) = obj.downcast_bound::<PyCuboid>(py) {
            Ok(self
                .scene
                .add_object(Box::new(obj.borrow().inner.clone()), transform.inner.clone(), mat))
        } else {
            Err(PyTypeError::new_err("invalid argument"))
        }
    }
}

#[pyclass]
#[pyo3(name = "ObjectTransform")]
struct PyObjectTransform {
    inner: ObjectTransform
}

#[pymethods]
impl PyObjectTransform {
    #[new]
    #[pyo3(signature = (translation, axis_angle=None))]
    fn new(translation: (f64, f64, f64), axis_angle: Option<((f64, f64, f64), f64)>) -> Self {
        //let rotation = axis_angle.map(|(axis, angle)| Quaternion::from_axis_angle(axis.into(), Rad(angle)) );
        let translation = Vector3::new(translation.0, translation.1, translation.2);
        let inner = match axis_angle {
            Some((axis, angle)) => {
                ObjectTransform::new(Quaternion::from_axis_angle(axis.into(), Rad(angle)), translation)
            }
            None => {
                ObjectTransform::with_translation(translation)      
            }
        };
        Self {
            inner
        }
    }
}

#[pyclass]
#[pyo3(name = "Sphere")]
struct PySphere {
    inner: Sphere,
}

#[pymethods]
impl PySphere {
    #[new]
    fn new(radius: f64) -> Self {
        Self {
            inner: Sphere::new(radius),
        }
    }
}

#[pyclass]
#[pyo3(name = "Plane")]
struct PyPlane {
    inner: Plane,
}

#[pymethods]
impl PyPlane {
    #[new]
    fn new() -> Self {
        Self {
            inner: Plane::new(),
        }
    }
}

#[pyclass]
#[pyo3(name = "Cuboid")]
struct PyCuboid {
    inner: Cuboid,
}

#[pymethods]
impl PyCuboid {
    #[new]
    fn new(length_x: f64, length_y: f64, length_z: f64) -> Self {
        Self {
            inner: Cuboid::new(length_x, length_y, length_z),
        }
    }
}

#[pyclass]
#[pyo3(name = "SimpleMaterial")]
struct PySimpleMaterial {
    inner: SimpleMaterial,
}

#[pymethods]
impl PySimpleMaterial {
    #[new]
    fn new(color: (f32, f32, f32)) -> Self {
        Self {
            inner: SimpleMaterial::new(color.into()),
        }
    }
}

#[pyclass]
#[pyo3(name = "CheckerboardMaterial")]
struct PyCheckerboardMaterial {
    inner: CheckerboardMaterial,
}

#[pymethods]
impl PyCheckerboardMaterial {
    #[new]
    fn new(color: (f32, f32, f32), direction: (f64, f64, f64)) -> Self {
        Self {
            inner: CheckerboardMaterial::new(color.into(), direction.into()),
        }
    }
}

#[pyclass]
#[pyo3(name = "FresnelMaterial")]
struct PyFresnelMaterial {
    inner: FresnelMaterial,
}

#[pymethods]
impl PyFresnelMaterial {
    #[new]
    fn new(index: f64, outer_index: f64) -> Self {
        Self {
            inner: FresnelMaterial::new(index, outer_index),
        }
    }
}

#[pyclass]
#[pyo3(name = "LinearGRINFresnelMaterial")]
struct PyLinearGRINFresnelMaterial {
    inner: LinearGRINFresnelMaterial,
}

#[pymethods]
impl PyLinearGRINFresnelMaterial {
    #[new]
    fn new(reference_index: f64, gradient: (f64, f64, f64), outer_index: f64) -> Self {
        Self {
            inner: LinearGRINFresnelMaterial::new(reference_index, gradient.into(), outer_index),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn grinray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPerspectiveCamera>()?;
    m.add_class::<PyPerspectiveCameraParameters>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyObjectTransform>()?;
    m.add_class::<PySphere>()?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PyCuboid>()?;
    m.add_class::<PySimpleMaterial>()?;
    m.add_class::<PyCheckerboardMaterial>()?;
    m.add_class::<PyFresnelMaterial>()?;
    m.add_class::<PyLinearGRINFresnelMaterial>()?;
    Ok(())
}
