use std::collections::HashMap;

use cgmath::{InnerSpace, Vector3};
use slotmap::{SlotMap, KeyData};

use crate::{RTIntersection, Ray, material::Material};

/// Minimum distance at which intersect_ray will detect an intersection
pub(crate) const RAYDIST_EPSILON: f64 = 1e-7;
/// Casting rays over extreme distances can introduce large numerical errors. To prevent these
/// errors from causing problems down the line, we allow the raytracer to consider all intersections
/// with a distance larger than this value to be invalid (i.e. the same as "no intersection")
pub(crate) const RAYDIST_MAX: f64 = 1e10;

#[derive(Hash,PartialEq,Eq,Clone, Copy)]
pub(crate) struct ObjectID(pub(crate) ObjectKey);
#[derive(Hash,PartialEq,Eq,Clone, Copy)]
pub(crate) struct MaterialID(pub(crate) MaterialKey);
type ObjectKey = slotmap::DefaultKey;
type MaterialKey = slotmap::DefaultKey;

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

pub(crate) struct Scene {
    objects: SlotMap<ObjectKey, Box<dyn RTObject + Send + Sync>>,
    materials: SlotMap<MaterialKey, Box<dyn Material + Send + Sync>>,
    object_materials: HashMap<ObjectKey, MaterialKey>,
    sky_color: Vector3<f32>
}

impl Scene {
    pub(crate) fn new(sky_color: Vector3<f32>) -> Self {
        Self { objects: SlotMap::new(), materials: SlotMap::new(), object_materials: HashMap::new(), sky_color }
    }

    pub(crate) fn cast_ray(&self, ray: &Ray) -> Option<(ObjectID, RTIntersection)> {
        let mut closest_intersection: Option<(ObjectKey, RTIntersection)> = None;
        for (object_key, object) in &self.objects {
            if let Some(intersection) = object.intersect_ray(ray) {
                if intersection.ray_dist > RAYDIST_MAX {
                    continue;
                }
                closest_intersection = match closest_intersection {
                    Some(prev_intersection) => {
                        if intersection.ray_dist < prev_intersection.1.ray_dist {
                            Some((object_key, intersection))
                        } else {
                            Some(prev_intersection)
                        }
                    }
                    None => Some((object_key, intersection)),
                }
            }
        }
        closest_intersection.map(|(object_key,intersection)| (ObjectID(object_key), intersection))
    }

    pub(crate) fn add_object(&mut self, obj: Box<dyn RTObject + Send + Sync>, mat: MaterialID) -> ObjectID {
        let object_key = self.objects.insert(obj);
        self.object_materials.insert(object_key, mat.0);
        ObjectID(object_key)
    }

    pub(crate) fn add_material(&mut self, mat: Box<dyn Material + Send + Sync>) -> MaterialID {
        MaterialID(self.materials.insert(mat))
    }

    pub(crate) fn get_object(&self, id: ObjectKey) -> Option<&(dyn RTObject + Send + Sync)> {
        self.objects.get(id).map(|b| b.as_ref())
    }

    pub(crate) fn get_object_material(&self, obj: ObjectID) -> &Box<dyn Material + Send + Sync> {
        let material_key = self.object_materials.get(&obj.0).unwrap();
        self.materials.get(*material_key).unwrap()
    }

    pub(crate) fn get_sky_color(&self) -> &Vector3<f32> {
        &self.sky_color
    }
}

/// Trait for objects that can be intersected by straight lines, i.e. objects that can be raytraced
pub trait RTObject {
    /// Closest intersection of a ray with this objects (but at least RAYDIST_EPSILON away from the
    /// start of the ray).
    /// 
    /// This function does *not* consider intersections that are "behind" the start of the ray
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection>;
    /// Closest intersection with a line.
    /// 
    /// This function also checks for intersections "behind" the start of the given ray and does
    /// *not* have a minimum distance between the ray start and the intersection point.
    fn intersect_line(&self, line: &Ray) -> Option<RTIntersection>;
    /// Some materials may introduce material properties with a spatial dependency. This point is
    /// used as a reference for these properties.
    fn reference_point(&self) -> &Vector3<f64>;
}

/// Sphere centered around a point with a given radius
#[derive(Clone)]
pub struct Sphere {
    pub center: Vector3<f64>,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl RTObject for Sphere {
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection> {
        let a = ray.start - self.center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        if discriminant >= 0.0 {
            let discriminant = discriminant.sqrt();
            let solution1 = -ad - discriminant;
            let solution2 = -ad + discriminant;
            let ray_dist = if solution1 >= RAYDIST_EPSILON {
                solution1
            }
            else if solution2 >= RAYDIST_EPSILON {
                solution2
            }
            else {
                return None
            };
            let point = ray.start + ray_dist * ray.dir;
            let normal = (point - self.center).normalize();
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        } else {
            None
        }
    }

    fn intersect_line(&self, ray: &Ray) -> Option<RTIntersection> {
        let a = ray.start - self.center;
        let ad = a.dot(ray.dir);
        let discriminant = ad * ad - a.dot(a) + self.radius * self.radius;
        if discriminant >= 0.0 {
            let discriminant = discriminant.sqrt();
            let solution1 = -ad - discriminant;
            let solution2 = -ad + discriminant;
            let ray_dist = if solution1.abs() < solution2.abs() {
                solution1
            }
            else {
                solution2
            };
            let point = ray.start + ray_dist * ray.dir;
            let normal = (point - self.center).normalize();
            Some(RTIntersection {
                ray_dist,
                point,
                normal,
            })
        } else {
            None
        }
    }

    fn reference_point(&self) -> &Vector3<f64> {
        &self.center
    }
}

/// Infinite plane defined by an origin and a normal direction
#[derive(Clone)]
pub struct Plane {
    origin: Vector3<f64>,
    normal: Vector3<f64>
}

impl Plane {
    pub fn new(origin: Vector3<f64>, normal: Vector3<f64>) -> Self {
        Self { origin, normal }
    }
}

impl RTObject for Plane {
    fn intersect_ray(&self, ray: &Ray) -> Option<RTIntersection> {
        // Discard rays that are too close to being parallel
        let nd = self.normal.dot(ray.dir);
        if nd.abs() == 0.0 {
            None
        }
        else {
            let ray_dist = (self.origin- ray.start).dot(self.normal) / nd;
            if ray_dist >= RAYDIST_EPSILON {
                let point = ray.start + ray_dist * ray.dir;
                let normal = self.normal;
                Some(RTIntersection {
                    ray_dist,
                    point,
                    normal,
                })
            } else {
                None
            }
        }
    }

    fn intersect_line(&self, _ray: &Ray) -> Option<RTIntersection> {
        todo!()
    }

    fn reference_point(&self) -> &Vector3<f64> {
        &self.origin
    }
}
