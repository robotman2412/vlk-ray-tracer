use std::{error::Error, sync::Arc};

use glam::{Vec3, Vec4};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use crate::scene::{Object, PhysProp, Scene, Skybox, Transform};

/// On-GPU vec3.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuVec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<Vec3> for GpuVec4 {
    fn from(value: Vec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: 0.0,
        }
    }
}

impl From<Vec4> for GpuVec4 {
    fn from(value: Vec4) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
        }
    }
}

/// On-GPU representation of an object type.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuObjectType {
    Sphere = 0,
    Plane,
}

/// On-GPU representation of an object's transform.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuTransform {
    pub matrix: [f32; 16],
    pub inv_matrix: [f32; 16],
}
unsafe impl Send for GpuTransform {}
unsafe impl Sync for GpuTransform {}

impl From<Transform> for GpuTransform {
    fn from(value: Transform) -> Self {
        Self {
            matrix: value.matrix().to_cols_array(),
            inv_matrix: value.inv_matrix().to_cols_array(),
        }
    }
}

/// On-GPU representation of an object's physical properties.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuPhysProp {
    pub ior: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub color: GpuVec4,
    pub emission: GpuVec4,
}
unsafe impl Send for GpuPhysProp {}
unsafe impl Sync for GpuPhysProp {}

impl From<PhysProp> for GpuPhysProp {
    fn from(value: PhysProp) -> Self {
        Self {
            ior: value.ior,
            opacity: value.opacity,
            roughness: value.roughness,
            color: value.color.into(),
            emission: value.emission.into(),
        }
    }
}

/// On-GPU representation of an object.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuObject {
    pub transform: GpuTransform,
    pub prop: GpuPhysProp,
    pub shader_type: u32,
}
unsafe impl Send for GpuObject {}
unsafe impl Sync for GpuObject {}

impl From<&(dyn Object + Send + Sync)> for GpuObject {
    fn from(value: &(dyn Object + Send + Sync)) -> Self {
        From::<&dyn Object>::from(value)
    }
}
impl From<&dyn Object> for GpuObject {
    fn from(value: &dyn Object) -> Self {
        Self {
            transform: value.transform().clone().into(),
            prop: value.phys_prop().clone().into(),
            shader_type: value.shader_type() as u32,
        }
    }
}

/// On-GPU representation of a skybox.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuSkybox {
    /// Ground color.
    pub ground_color: GpuVec4,
    /// Horizon color.
    pub horizon_color: GpuVec4,
    /// Skybox color.
    pub skybox_color: GpuVec4,
    /// Sun color.
    pub sun_color: GpuVec4,
    /// Unit vector pointing at the sun.
    pub sun_direction: GpuVec4,
    /// Dot product threshold for a ray to be pointing at the sun.
    pub sun_radius: f32,
}
unsafe impl Send for GpuSkybox {}
unsafe impl Sync for GpuSkybox {}

impl From<Skybox> for GpuSkybox {
    fn from(value: Skybox) -> Self {
        Self {
            ground_color: value.ground_color.into(),
            horizon_color: value.horizon_color.into(),
            skybox_color: value.skybox_color.into(),
            sun_color: value.sun_color.into(),
            sun_direction: value.sun_direction.into(),
            sun_radius: value.sun_radius,
        }
    }
}

/// On-GPU representation of a scene.
/// Unlike the others, this is not a single bufferable object, but a collection of buffers.
pub struct GpuScene {
    pub objects: Subbuffer<[GpuObject]>,
    pub object_count: u32,
    pub skybox: Subbuffer<[GpuSkybox]>,
}

impl GpuScene {
    pub fn build(
        allocator: Arc<dyn MemoryAllocator>,
        scene: &Scene,
    ) -> Result<Self, Box<dyn Error>> {
        let objects = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            scene.objects.iter().map(|object| object.as_ref().into()),
        )?;
        let skybox = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [GpuSkybox::from(scene.skybox.clone())].into_iter(),
        )?;
        Ok(Self {
            objects,
            object_count: scene.objects.len() as u32,
            skybox,
        })
    }
}
