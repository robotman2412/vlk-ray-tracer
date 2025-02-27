use std::{error::Error, fmt::Debug, ops::Deref, sync::Arc, u32};

use glam::{Vec2, Vec3, Vec4};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use crate::{mesh::*, scene::*};

/// On-GPU vec3.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, BufferContents)]
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

/// On-GPU Vec2.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C, align(8))]
#[derive(Debug, Copy, Clone, BufferContents)]
pub struct GpuVec2 {
    pub x: f32,
    pub y: f32,
}

impl From<Vec2> for GpuVec2 {
    fn from(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

/// On-GPU representation of an object type.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuObjectType {
    Sphere = 0,
    Plane,
    Mesh,
}

/// On-GPU representation of an object's transform.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Debug, Copy, Clone, BufferContents)]
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
#[derive(Debug, Copy, Clone, BufferContents)]
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
#[derive(Debug, Copy, Clone, BufferContents)]
pub struct GpuObject {
    pub transform: GpuTransform,
    pub prop: GpuPhysProp,
    pub model_type: u32,
    pub model_index: u32,
}
unsafe impl Send for GpuObject {}
unsafe impl Sync for GpuObject {}

/// On-GPU representation of a skybox.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Debug, Copy, Clone, BufferContents)]
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

/// On-GPU representation of a model.
/// WARNING: The GPU types can be smaller than their alignment, so the most-aligned field must always come last.
#[repr(C)]
#[derive(Debug, Copy, Clone, BufferContents)]
pub struct GpuMesh {
    /// Number of triangles in the model.
    pub num_tris: u32,
    /// BVH root node's offset into the BVH buffer, or -1 if none.
    pub bvh_offset: u32,
    /// Triangles' offset into the triangle buffer.
    /// Triangles are a uvec3 representing the vertex index.
    pub tri_offset: u32,
    /// Vertices' offset into the vertex buffer.
    /// Vertices are a vec3 representing object space position.
    pub vert_offset: u32,
    /// Normals' offset into the normal buffer, or -1 if none.
    /// Normals are a vec3 representing the vertex normal in object space.
    pub norm_offset: u32,
    /// Vertex colors' offset into the vertex color buffer, or -1 if none.
    /// Vertex colors are a vec3.
    pub vcol_offset: u32,
    /// UV coordinates' offset into the UVs buffer, or -1 if none.
    /// UV coordinates are a vec2.
    pub uv_offset: u32,
}
unsafe impl Send for GpuMesh {}
unsafe impl Sync for GpuMesh {}

/// On-GPU representation of a BVH.
#[repr(C)]
#[derive(Debug, Copy, Clone, BufferContents)]
pub struct GpuBvh {
    /// Minimum position.
    pub min: GpuVec4,
    /// Maximum position.
    pub max: GpuVec4,
    /// Index of child nodes or vertices.
    pub children: u32,
    /// Number of vertices; 0 means it has 2 BVH node children.
    pub tri_count: u32,
}
unsafe impl Send for GpuBvh {}
unsafe impl Sync for GpuBvh {}

/// On-GPU representation of a scene.
/// Unlike the others, this is not a single bufferable object, but a collection of buffers.
pub struct GpuScene {
    pub objects: Subbuffer<[GpuObject]>,
    pub object_count: u32,
    pub meshes: Subbuffer<[GpuMesh]>,
    pub tris: Subbuffer<[u32]>,
    pub verts: Subbuffer<[GpuVec4]>,
    pub norms: Subbuffer<[GpuVec4]>,
    pub vcols: Subbuffer<[GpuVec4]>,
    pub uvs: Subbuffer<[GpuVec2]>,
    pub skybox: Subbuffer<[GpuSkybox]>,
    pub bvh: Subbuffer<[GpuBvh]>,
}

impl Debug for GpuScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuScene")
            .field("objects", &self.objects.read().unwrap().deref())
            .field("object_count", &self.object_count)
            .field("meshes", &self.meshes.read().unwrap().deref())
            .field("tris", &self.tris.read().unwrap().deref())
            .field("verts", &self.verts.read().unwrap().deref())
            .field("norms", &self.norms.read().unwrap().deref())
            .field("vcols", &self.vcols.read().unwrap().deref())
            .field("uvs", &self.uvs.read().unwrap().deref())
            .field("bvh", &self.bvh.read().unwrap().deref())
            .field("skybox", &self.skybox.read().unwrap().deref())
            .finish()
    }
}

#[derive(Default)]
struct NodeBuildCtx {
    objects: Vec<GpuObject>,
    meshes: Vec<GpuMesh>,
    tris: Vec<u32>,
    verts: Vec<GpuVec4>,
    norms: Vec<GpuVec4>,
    vcols: Vec<GpuVec4>,
    uvs: Vec<GpuVec2>,
    bvh: Vec<GpuBvh>,
}

impl GpuScene {
    fn build_bvh(out: &mut NodeBuildCtx, node: &Bvh, index: usize, tri_offset: u32) {
        match &node.content {
            BvhContent::Node(val) => {
                out.bvh.push(GpuBvh {
                    min: val.0.min.into(),
                    max: val.0.max.into(),
                    children: 0,
                    tri_count: 0,
                });
                out.bvh.push(GpuBvh {
                    min: val.1.min.into(),
                    max: val.1.max.into(),
                    children: 0,
                    tri_count: 0,
                });
                out.bvh[index].children = index as u32 + 1;
                let child_index = out.bvh.len() - 2;
                Self::build_bvh(out, &val.0, child_index, tri_offset);
                Self::build_bvh(out, &val.1, child_index + 1, tri_offset);
            }
            BvhContent::Leaf(leaf) => {
                out.bvh[index].children = leaf.begin as u32 + tri_offset;
                out.bvh[index].tri_count = (leaf.end - leaf.begin) as u32;
            }
        }
    }

    fn build_mesh(out: &mut NodeBuildCtx, mesh: &Mesh) {
        let mut gpu_mesh = GpuMesh {
            num_tris: mesh.tris.len() as u32,
            bvh_offset: u32::MAX,
            tri_offset: (out.tris.len() / 3) as u32,
            vert_offset: out.verts.len() as u32,
            norm_offset: u32::MAX,
            vcol_offset: u32::MAX,
            uv_offset: u32::MAX,
        };
        out.tris.reserve(mesh.tris.len() * 3);
        for tri in &mesh.tris {
            for corner in tri {
                out.tris.push(*corner as u32);
            }
        }
        out.verts
            .extend(mesh.verts.iter().map(|f| GpuVec4::from(*f)));
        if let Some(bvh) = mesh.bvh.as_ref() {
            gpu_mesh.bvh_offset = out.bvh.len() as u32;
            out.bvh.push(GpuBvh {
                min: bvh.min.into(),
                max: bvh.max.into(),
                children: 0,
                tri_count: 0,
            });
            println!("Converting BVH to GPU format...");
            Self::build_bvh(out, bvh, out.bvh.len() - 1, gpu_mesh.tri_offset);
            println!("Done! Created {} entries", out.bvh.len());
        }
        if let Some(normals) = mesh.normals.as_ref() {
            gpu_mesh.norm_offset = out.norms.len() as u32;
            out.norms.extend(normals.iter().map(|f| GpuVec4::from(*f)));
        }
        if let Some(vcols) = mesh.vert_cols.as_ref() {
            gpu_mesh.vcol_offset = out.vcols.len() as u32;
            out.vcols.extend(vcols.iter().map(|f| GpuVec4::from(*f)));
        }
        if let Some(uvs) = mesh.vert_uv.as_ref() {
            gpu_mesh.uv_offset = out.uvs.len() as u32;
            out.uvs.extend(uvs.iter().map(|f| GpuVec2::from(*f)));
        }
        out.meshes.push(gpu_mesh);
    }

    fn build_node(out: &mut NodeBuildCtx, transform: Transform, node: &Node) -> GpuObject {
        let (model_type, model_index) = match &node.model {
            Model::None => unreachable!(),
            Model::Sphere => (GpuObjectType::Sphere, 0),
            Model::Plane => (GpuObjectType::Plane, 0),
            Model::Mesh(mesh) => {
                let index = out.meshes.len();
                Self::build_mesh(out, mesh);
                (GpuObjectType::Mesh, index)
            }
        };
        GpuObject {
            transform: (node.transform * transform).into(),
            prop: node.prop.into(),
            model_type: model_type as u32,
            model_index: model_index as u32,
        }
    }

    fn build_nodes(out: &mut NodeBuildCtx, transform: Transform, nodes: &[Node]) {
        for node in nodes {
            if node.model != Model::None {
                let tmp = Self::build_node(out, transform, node);
                out.objects.push(tmp);
            }
            Self::build_nodes(out, transform * node.transform, &node.children);
        }
    }

    pub fn build(
        allocator: Arc<dyn MemoryAllocator>,
        scene: &Scene,
    ) -> Result<Self, Box<dyn Error>> {
        let mut ctx = NodeBuildCtx::default();
        Self::build_nodes(&mut ctx, Default::default(), &scene.nodes);

        let buf_info = BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        };

        // TODO: Vulkano doesn't support bindless yet.
        if ctx.meshes.is_empty() {
            ctx.meshes.push(GpuMesh {
                num_tris: 0,
                bvh_offset: 0,
                tri_offset: 0,
                vert_offset: 0,
                norm_offset: 0,
                vcol_offset: 0,
                uv_offset: 0,
            });
        }
        if ctx.tris.is_empty() {
            ctx.tris.push(0);
        }
        if ctx.verts.is_empty() {
            ctx.verts.push(Vec4::splat(0.0).into());
        }
        if ctx.norms.is_empty() {
            ctx.norms.push(Vec4::splat(0.0).into());
        }
        if ctx.vcols.is_empty() {
            ctx.vcols.push(Vec4::splat(0.0).into());
        }
        if ctx.uvs.is_empty() {
            ctx.uvs.push(Vec2::splat(0.0).into());
        }
        if ctx.bvh.is_empty() {
            ctx.bvh.push(GpuBvh {
                min: Vec4::splat(0.0).into(),
                max: Vec4::splat(0.0).into(),
                children: 0,
                tri_count: 0,
            });
        }

        let objects = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.objects.into_iter(),
        )?;
        let meshes = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.meshes.into_iter(),
        )?;
        let tris = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.tris.into_iter(),
        )?;
        let verts = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.verts.into_iter(),
        )?;
        let norms = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.norms.into_iter(),
        )?;
        let vcols = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.vcols.into_iter(),
        )?;
        let uvs = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.uvs.into_iter(),
        )?;
        let bvh = Buffer::from_iter(
            allocator.clone(),
            buf_info.clone(),
            alloc_info.clone(),
            ctx.bvh.into_iter(),
        )?;
        let skybox = Buffer::from_iter(
            allocator,
            buf_info.clone(),
            alloc_info.clone(),
            [GpuSkybox::from(scene.skybox)].into_iter(),
        )?;

        Ok(Self {
            objects,
            object_count: scene.nodes.len() as u32,
            skybox,
            meshes,
            tris,
            verts,
            norms,
            vcols,
            uvs,
            bvh,
        })
    }
}
