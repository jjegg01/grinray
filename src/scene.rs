use std::collections::HashMap;

use slotmap::SlotMap;

use crate::{materials::Material, objects::{ObjectTransform, RTObject}, RTIntersection, Ray, Tracer, RAYDIST_MAX};

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct ObjectID(pub(crate) ObjectKey);
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct MaterialID(pub(crate) MaterialKey);
pub(crate) type ObjectKey = slotmap::DefaultKey;
pub(crate) type MaterialKey = slotmap::DefaultKey;

pub struct Scene<T: Tracer> {
    objects: SlotMap<ObjectKey, (Box<dyn RTObject + Send + Sync>, ObjectTransform)>,
    materials: SlotMap<MaterialKey, Box<dyn Material<T> + Send + Sync>>,
    object_materials: HashMap<ObjectKey, MaterialKey>,
}

impl<T: Tracer> Scene<T> {
    pub fn new() -> Self {
        Self {
            objects: SlotMap::new(),
            materials: SlotMap::new(),
            object_materials: HashMap::new(),
        }
    }

    pub(crate) fn cast_ray(&self, ray: &Ray) -> Option<(ObjectID, RTIntersection)> {
        let mut closest_intersection: Option<(ObjectKey, RTIntersection)> = None;
        for (object_key, (object, object_transform)) in &self.objects {
            if let Some(intersection) = object.intersect_ray(object_transform, ray) {
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
        closest_intersection.map(|(object_key, intersection)| (ObjectID(object_key), intersection))
    }

    pub fn add_object(
        &mut self,
        obj: Box<dyn RTObject + Send + Sync>,
        transform: ObjectTransform,
        mat: MaterialID,
    ) -> ObjectID {
        let object_key = self.objects.insert((obj, transform));
        self.object_materials.insert(object_key, mat.0);
        ObjectID(object_key)
    }

    pub fn add_material(&mut self, mat: Box<dyn Material<T> + Send + Sync>) -> MaterialID {
        MaterialID(self.materials.insert(mat))
    }

    pub(crate) fn get_object(&self, id: ObjectKey) -> Option<(&(dyn RTObject + Send + Sync), &ObjectTransform)> {
        self.objects.get(id).map(|b| (b.0.as_ref(), &b.1))
    }

    pub(crate) fn get_object_material(&self, obj: ObjectID) -> &Box<dyn Material<T> + Send + Sync> {
        let material_key = self.object_materials.get(&obj.0).unwrap();
        self.materials.get(*material_key).unwrap()
    }
}
