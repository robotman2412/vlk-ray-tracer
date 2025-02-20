use glam::Mat4;
use glam::Vec3;

use crate::shader_buffer::GpuObjectType;

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct Transform {
    matrix: Mat4,
    inv_matrix: Mat4,
}
impl Eq for Transform {}

impl Transform {
    pub fn matrix<'a>(&'a self) -> &'a Mat4 {
        &self.matrix
    }
    pub fn inv_matrix<'a>(&'a self) -> &'a Mat4 {
        &self.inv_matrix
    }
    pub fn set_matrix(&mut self, matrix: Mat4) {
        self.matrix = matrix;
        self.inv_matrix = matrix.inverse();
    }
    pub fn ray_world_to_local(&self, ray: Ray) -> Ray {
        Ray {
            pos: self.inv_matrix.transform_point3(ray.pos),
            normal: self.inv_matrix.transform_vector3(ray.normal),
        }
    }
    pub fn ray_local_to_world(&self, ray: Ray) -> Ray {
        Ray {
            pos: self.matrix.transform_point3(ray.pos),
            normal: self.matrix.transform_vector3(ray.normal),
        }
    }
    pub fn normal_world_to_local(&self, normal: Vec3) -> Vec3 {
        self.inv_matrix.transform_vector3(normal)
    }
    pub fn normal_local_to_world(&self, normal: Vec3) -> Vec3 {
        self.matrix.transform_vector3(normal)
    }
    pub fn local_to_world(&self, pos: Vec3) -> Vec3 {
        self.matrix.transform_point3(pos)
    }
    pub fn world_to_local(&self, pos: Vec3) -> Vec3 {
        self.inv_matrix.transform_point3(pos)
    }
}

impl From<Mat4> for Transform {
    fn from(matrix: Mat4) -> Transform {
        Transform {
            matrix,
            inv_matrix: matrix.inverse(),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Ray {
    /// Position of the ray.
    pub pos: Vec3,
    /// Direction the ray is facing.
    pub normal: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct PhysProp {
    pub ior: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub color: Vec3,
    pub emission: Vec3,
}
impl Eq for PhysProp {}

impl PhysProp {
    pub fn from_color(color: Vec3) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity: 1.0,
            roughness: 1.0,
            color,
            emission: Vec3::ZERO,
        }
    }
    pub fn from_opacity(color: Vec3, opacity: f32) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity,
            roughness: 1.0,
            color,
            emission: Vec3::ZERO,
        }
    }
    pub fn from_emission(color: Vec3, emission: Vec3) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity: 1.0,
            roughness: 1.0,
            color,
            emission,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Intersect {
    /// Intersection position in world space.
    pub pos: Vec3,
    /// Surface normal.
    pub normal: Vec3,
    /// Physical properties at the intersection.
    pub prop: PhysProp,
    /// Distance from the ray origin.
    pub distance: f32,
    /// Whether the ray started outside the objct.
    pub is_entry: bool,
}
impl Eq for Intersect {}

pub trait Object {
    fn phys_prop<'a>(&'a self) -> &'a PhysProp;
    fn phys_prop_mut<'a>(&'a mut self) -> &'a mut PhysProp;
    fn set_phys_prop(&mut self, prop: PhysProp) {
        *self.phys_prop_mut() = prop;
    }
    fn transform<'a>(&'a self) -> &'a Transform;
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform;
    fn set_transform(&mut self, pos: Transform) {
        *self.transform_mut() = pos;
    }
    /// Perform an intersection test with a ray in world space.
    fn intersect(&self, ray: &Ray) -> Option<Intersect>;
    /// Get the object type to be sent to the RT shader.
    fn shader_type(&self) -> GpuObjectType;
}

pub struct Sphere {
    pub transform: Transform,
    pub prop: PhysProp,
}

impl Object for Sphere {
    fn phys_prop<'a>(&'a self) -> &'a PhysProp {
        &self.prop
    }
    fn phys_prop_mut<'a>(&'a mut self) -> &'a mut PhysProp {
        &mut self.prop
    }
    fn transform<'a>(&'a self) -> &'a Transform {
        &self.transform
    }
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform {
        &mut self.transform
    }

    fn intersect(&self, ray: &Ray) -> Option<Intersect> {
        let ray = self.transform.ray_world_to_local(*ray);
        let a = -ray.normal.dot(ray.pos);
        let b = a * a - ray.pos.length_squared() + 1.0;

        if b < 0.0 {
            return None;
        }
        let distance: f32;
        if b < 0.00000001 {
            if a > 0.00000001 {
                distance = a;
            } else {
                return None;
            }
        } else {
            let dist0 = a + b.sqrt();
            let dist1 = a - b.sqrt();
            if dist0 < dist1 && dist0 > 0.00000001 {
                distance = dist0;
            } else if dist1 > 0.00000001 {
                distance = dist1;
            } else {
                return None;
            }
        };
        let pos = ray.pos + ray.normal * distance;

        return Some(Intersect {
            pos: self.transform.local_to_world(pos),
            normal: self.transform.normal_local_to_world(pos),
            prop: self.prop,
            distance,
            is_entry: ray.pos.length_squared() > 1.0,
        });
    }

    fn shader_type(&self) -> GpuObjectType {
        GpuObjectType::Sphere
    }
}

pub struct Plane {
    pub transform: Transform,
    pub prop: PhysProp,
}

impl Object for Plane {
    fn phys_prop<'a>(&'a self) -> &'a PhysProp {
        &self.prop
    }
    fn phys_prop_mut<'a>(&'a mut self) -> &'a mut PhysProp {
        &mut self.prop
    }
    fn transform<'a>(&'a self) -> &'a Transform {
        &self.transform
    }
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform {
        &mut self.transform
    }

    fn intersect(&self, ray: &Ray) -> Option<Intersect> {
        let ray = self.transform.ray_world_to_local(*ray);
        if ray.normal[2].abs() < 0.00000001 {
            return None;
        }
        let distance = -ray.pos[2] / ray.normal[2];
        if distance < 0.00000001 {
            return None;
        }
        let pos = ray.pos + ray.normal * distance;
        if pos[0].abs() > 1.0 || pos[1].abs() > 1.0 {
            return None;
        }
        Some(Intersect {
            pos: self.transform.local_to_world(pos),
            normal: self
                .transform
                .normal_local_to_world(Vec3::new(0.0, 0.0, ray.pos[2].signum())),
            prop: self.prop,
            distance,
            is_entry: true,
        })
    }

    fn shader_type(&self) -> GpuObjectType {
        GpuObjectType::Plane
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Skybox {
    /// Ground color.
    pub ground_color: Vec3,
    /// Horizon color.
    pub horizon_color: Vec3,
    /// Skybox color.
    pub skybox_color: Vec3,
    /// Sun color.
    pub sun_color: Vec3,
    /// Unit vector pointing at the sun.
    pub sun_direction: Vec3,
    /// Dot product threshold for a ray to be pointing at the sun.
    pub sun_radius: f32,
}

impl Skybox {
    pub fn empty() -> Skybox {
        Skybox {
            ground_color: Vec3::ZERO,
            horizon_color: Vec3::ZERO,
            skybox_color: Vec3::ZERO,
            sun_color: Vec3::ZERO,
            sun_direction: Vec3::NEG_Y,
            sun_radius: 1.0,
        }
    }
}

impl Default for Skybox {
    fn default() -> Self {
        Self {
            ground_color: Vec3::new(0.3, 0.15, 0.075),
            horizon_color: Vec3::new(0.7, 0.9, 1.0),
            skybox_color: Vec3::new(0.0, 0.7, 0.8),
            sun_color: Vec3::new(2.0, 2.0, 1.4),
            sun_direction: Vec3::new(0.577350269, -0.577350269, -0.577350269),
            sun_radius: 0.8,
        }
    }
}

pub struct Scene {
    /// List of objects in the scene.
    pub objects: Vec<Box<dyn Object + Send + Sync>>,
    /// Scene skybox.
    pub skybox: Skybox,
}

impl Scene {
    pub fn empty() -> Scene {
        Scene {
            objects: Vec::new(),
            skybox: Skybox::default(),
        }
    }
}
