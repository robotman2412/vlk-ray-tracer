use std::ops::Mul;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use obj::Obj;

use crate::mesh::*;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Transform {
    matrix: Mat4,
    inv_matrix: Mat4,
}
impl Eq for Transform {}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Self::Output {
        Self {
            matrix: self.matrix * rhs.matrix,
            inv_matrix: rhs.inv_matrix * self.inv_matrix,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysProp {
    pub ior: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub color: Vec3,
    pub emission: Vec3,
}
impl Eq for PhysProp {}

impl Default for PhysProp {
    fn default() -> Self {
        Self {
            ior: 1.5,
            opacity: 1.0,
            roughness: 0.5,
            color: Vec3::new(0.5, 0.5, 0.5),
            emission: Vec3::ZERO,
        }
    }
}

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

#[derive(Debug, Clone)]
pub enum Model {
    /// Node has no model.
    None,
    /// Sphere with radius 1.
    Sphere,
    /// XY-plane square with radius 1.
    Plane,
    /// Mesh made out of triangles.
    Mesh(Arc<Mesh>),
}
impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // If they point to the same mesh, the model is considered the same.
            (Self::Mesh(l0), Self::Mesh(r0)) => {
                l0.as_ref() as *const Mesh == r0.as_ref() as *const Mesh
            }
            // All other variants do not have a value.
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct Node {
    /// Node's position, rotation and scale.
    pub transform: Transform,
    /// Child nodes.
    pub children: Vec<Node>,
    /// Node's model.
    pub model: Model,
    /// Node's material/properties.
    pub prop: PhysProp,
}

impl From<&Obj> for Node {
    fn from(value: &Obj) -> Self {
        let mut groups = vec![];

        for object in &value.data.objects {
            groups.extend(object.groups.iter());
        }

        if groups.len() == 1 {
            Self {
                model: Model::Mesh(Arc::new(Mesh::from_group(value, groups[0]))),
                ..Default::default()
            }
        } else {
            Self {
                children: groups
                    .iter()
                    .map(|group| Self {
                        model: Model::Mesh(Arc::new(Mesh::from_group(value, group))),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
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
            ground_color: Vec3::new(0.15, 0.075, 0.0375),
            horizon_color: Vec3::new(0.35, 0.45, 0.5),
            skybox_color: Vec3::new(0.0, 0.35, 0.4),
            sun_color: Vec3::splat(16.0),
            sun_direction: Vec3::new(0.577350269, -0.577350269, -0.577350269),
            sun_radius: 0.9,
        }
    }
}

pub struct Scene {
    /// Scene root node.
    pub nodes: Vec<Node>,
    /// Scene skybox.
    pub skybox: Skybox,
}
