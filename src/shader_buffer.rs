use std::sync::Arc;

use vulkano::{
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    Validated,
};

use crate::scene::{Object, PhysProp, Scene, Skybox, Transform};

/// On-GPU representation of an object type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuObjectType {
    Sphere = 0,
    Plane,
}

/// On-GPU representation of an object's transform.
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
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuPhysProp {
    pub ior: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub color: [f32; 3],
    pub emission: [f32; 3],
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
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct GpuSkybox {
    /// Ground color.
    pub ground_color: [f32; 3],
    /// Horizon color.
    pub horizon_color: [f32; 3],
    /// Skybox color.
    pub skybox_color: [f32; 3],
    /// Sun color.
    pub sun_color: [f32; 3],
    /// Unit vector pointing at the sun.
    pub sun_direction: [f32; 3],
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
    pub skybox: GpuSkybox,
}

impl GpuScene {
    pub fn build(
        allocator: Arc<dyn MemoryAllocator>,
        scene: &Scene,
    ) -> Result<Self, Validated<AllocateBufferError>> {
        let objects = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            scene.objects.iter().map(|object| object.as_ref().into()),
        )?;
        Ok(Self {
            objects,
            skybox: scene.skybox.into(),
        })
    }
}
