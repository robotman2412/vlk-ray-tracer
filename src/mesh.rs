use core::f32;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use glam::{Vec2, Vec3};
use obj::{Group, IndexTuple, Obj};

/// Extra info to use while building the BVH.
#[derive(Debug, Clone, Copy)]
struct BvhTriAux {
    pub min: Vec3,
    pub max: Vec3,
    pub center: Vec3,
    pub area: f32,
}

/// Bounding-volume hierarchy.
#[derive(Debug, Clone)]
pub struct Bvh {
    pub min: Vec3,
    pub max: Vec3,
    pub content: BvhContent,
}

impl Bvh {
    pub const MAX_DEPTH: usize = 32;
    pub const MIN_TRI: usize = 2;
    pub const MAX_SLICES: usize = 5;

    /// Calculate [min, max] bounds for a range of triangles.
    fn calc_bounds(aux: &[BvhTriAux], begin: usize, end: usize) -> (Vec3, Vec3) {
        aux[begin..end]
            .iter()
            .map(|f| (f.min, f.max))
            .fold((Vec3::MAX, Vec3::MIN), |(min_a, max_a), (min_b, max_b)| {
                (min_a.min(min_b), max_a.max(max_b))
            })
    }

    /// Evaluate the surface area heuristic at a single point.
    fn eval_sah(aux: &[BvhTriAux], axis: usize, pos: f32, before: bool) -> f32 {
        let mut min = Vec3::MAX;
        let mut max = Vec3::MIN;
        let mut tri_area = 0.0;

        for tri in aux {
            if (tri.center[axis] < pos) == before {
                min = min.min(tri.min);
                max = max.max(tri.max);
                tri_area += tri.area;
            }
        }

        let box_size = max - min;
        let box_area = 2.0 * (box_size.x * (box_size.y + box_size.z) + box_size.y * box_size.z);

        box_area / tri_area
    }

    /// Get heuristic cost for splitting along an axis.
    fn eval_axis(&self, aux: &[BvhTriAux], axis: usize) -> (f32, f32, f32) {
        let range = self.content.as_leaf().unwrap();
        let slice_points: Vec<_> = {
            if range.end - range.begin <= Self::MAX_SLICES {
                aux[range.begin..range.end]
                    .iter()
                    .map(|f| f.center[axis])
                    .collect()
            } else {
                let scale = (self.max - self.min)[axis] / Self::MAX_SLICES as f32;
                (0..Self::MAX_SLICES)
                    .map(|f| (f as f32 + 0.5) * scale)
                    .collect()
            }
        };

        let mut best = (0f32, f32::MAX, f32::MAX);

        for point in slice_points {
            let cost0 = Self::eval_sah(&aux[range.begin..range.end], axis, point, true);
            let cost1 = Self::eval_sah(&aux[range.begin..range.end], axis, point, false);
            if cost0 + cost1 < best.1 + best.2 {
                best = (point, cost0, cost1);
            }
        }

        best
    }

    /// Split along an axis.
    fn try_split(
        &mut self,
        mesh: &mut Mesh,
        aux: &mut [BvhTriAux],
        axis: usize,
        pos: f32,
        cost: (f32, f32),
    ) -> bool {
        if let BvhContent::Leaf(data) = &self.content {
            // Index of first triangle greater than the threshold.
            let mut midpoint: Option<usize> = None;

            // Renumber triangles along the split.
            for i in data.begin..data.end {
                if aux[i].center[axis] > pos {
                    if midpoint == None {
                        midpoint = Some(i);
                    }
                } else if let Some(point) = midpoint {
                    // Triangle must be swapped to before the midpoint.
                    (mesh.tris[point], mesh.tris[i]) = (mesh.tris[i], mesh.tris[point]);
                    (aux[point], aux[i]) = (aux[i], aux[point]);
                    midpoint = Some(point + 1);
                }
            }
            if midpoint == None {
                return false;
            }

            // Triangles re-ordered, the node can now be split.
            self.content = BvhContent::Node((
                Box::new({
                    let (min, max) = Bvh::calc_bounds(aux, data.begin, midpoint.unwrap());
                    Bvh {
                        min,
                        max,
                        content: BvhContent::Leaf(BvhLeaf {
                            begin: data.begin,
                            end: midpoint.unwrap(),
                            cost: cost.0,
                        }),
                    }
                }),
                Box::new({
                    let (min, max) = Bvh::calc_bounds(aux, midpoint.unwrap(), data.end);
                    Bvh {
                        min,
                        max,
                        content: BvhContent::Leaf(BvhLeaf {
                            begin: midpoint.unwrap(),
                            end: data.end,
                            cost: cost.1,
                        }),
                    }
                }),
            ));
            return true;
        }
        false
    }

    /// Split a BVH node if possible.
    fn build_impl(&mut self, mesh: &mut Mesh, aux: &mut [BvhTriAux], depth: usize) {
        if let BvhContent::Leaf(data) = &self.content {
            // Limit condition.
            if data.end - data.begin <= 2 * Bvh::MIN_TRI || depth >= Bvh::MAX_DEPTH {
                return;
            }
        } else {
            return;
        }

        // Evaluate how good it would be to split along each axis.
        let (x_pos, x_cost0, x_cost1) = self.eval_axis(aux, 0);
        let (y_pos, y_cost0, y_cost1) = self.eval_axis(aux, 1);
        let (z_pos, z_cost0, z_cost1) = self.eval_axis(aux, 2);
        let x = x_cost0 + x_cost1;
        let y = y_cost0 + y_cost1;
        let z = z_cost0 + z_cost1;

        // Split along the axis with least cost.
        if x.is_finite() && x < y && x < z {
            self.try_split(mesh, aux, 0, x_pos, (x_cost0, x_cost1));
        } else if y.is_finite() && y < z {
            self.try_split(mesh, aux, 1, y_pos, (y_cost0, y_cost1));
        } else if z.is_finite() {
            self.try_split(mesh, aux, 2, z_pos, (z_cost0, z_cost1));
        }

        // Recursively split child nodes.
        if let BvhContent::Node(node) = &mut self.content {
            node.0.build_impl(mesh, aux, depth + 1);
            node.1.build_impl(mesh, aux, depth + 1);
        }
    }

    /// Build a BVH for the mesh, potentially changing the order of the triangles.
    fn build(mesh: &mut Mesh) -> Bvh {
        // Create auxiliary data.
        let mut aux: Box<[BvhTriAux]> = mesh
            .tris
            .iter()
            .map(|tri| {
                let a = mesh.verts[tri[0]];
                let b = mesh.verts[tri[1]];
                let c = mesh.verts[tri[2]];
                BvhTriAux {
                    min: a.min(b.min(c)),
                    max: a.max(b.max(c)),
                    center: (a + b + c) * 0.33333332,
                    area: (b - a).cross(c - a).length() * 0.5,
                }
            })
            .collect();

        // Create BVH root node.
        let mut tmp = Bvh {
            min: aux.iter().map(|f| f.min).fold(Vec3::MAX, |a, b| a.min(b)),
            max: aux.iter().map(|f| f.min).fold(Vec3::MIN, |a, b| a.max(b)),
            content: BvhContent::Leaf(BvhLeaf {
                begin: 0,
                end: mesh.tris.len(),
                cost: Bvh::eval_sah(&aux, 0, f32::MAX, true),
            }),
        };

        // Recursively try to split the BVH.
        tmp.build_impl(mesh, &mut aux, 0);

        tmp
    }
}

/// Inner node in a bounding-volume hierarchy.
#[derive(Debug, Clone)]
pub enum BvhContent {
    Node((Box<Bvh>, Box<Bvh>)),
    Leaf(BvhLeaf),
}

impl BvhContent {
    pub fn as_node<'a>(&'a self) -> Option<&'a (Box<Bvh>, Box<Bvh>)> {
        match self {
            Self::Node(val) => Some(val),
            _ => None,
        }
    }

    pub fn as_leaf<'a>(&'a self) -> Option<&'a BvhLeaf> {
        match self {
            Self::Leaf(val) => Some(val),
            _ => None,
        }
    }
}

/// Leaf node in a bounding-volume hierarchy.
#[derive(Debug, Clone)]
pub struct BvhLeaf {
    pub begin: usize,
    pub end: usize,
    pub cost: f32,
}

/// 3D mesh for use by the ray tracer..
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Bounding volume hierarchy.
    pub bvh: Option<Bvh>,
    /// Triangle vertices.
    pub tris: Vec<[usize; 3]>,
    /// Vertex positions.
    pub verts: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Option<Vec<Vec3>>,
    /// Vertex colors.
    pub vert_cols: Option<Vec<Vec3>>,
    /// Vertex UV coordinates.
    pub vert_uv: Option<Vec<Vec2>>,
}

impl Mesh {
    /// Create / update the BVH for this mesh.
    pub fn create_bvh(&mut self) {
        self.bvh = Some(Bvh::build(self));
    }
}

/// Temporary type used to deduplicate vertices.
#[derive(Debug, Clone, Copy)]
struct ObjPolyCorner {
    pos: usize,
    normal: Option<usize>,
    uv: Option<usize>,
    index: usize,
}

impl PartialEq for ObjPolyCorner {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.normal == other.normal && self.uv == other.uv
    }
}

impl Hash for ObjPolyCorner {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.hash(state);
        self.normal.hash(state);
        self.uv.hash(state);
    }
}
impl Eq for ObjPolyCorner {}

impl Mesh {
    /// Create a mesh from an Obj group.
    pub fn from_group(object: &Obj, group: &Group) -> Self {
        let mut tris = Vec::<[usize; 3]>::new();
        let mut verts = HashSet::<ObjPolyCorner>::new();
        let mut index = 0usize;
        let mut use_norm = false;
        let mut use_uv = false;

        let mut dedup_corner = |corner: &IndexTuple| {
            let corner = ObjPolyCorner {
                pos: corner.0,
                uv: corner.1,
                normal: corner.2,
                index,
            };
            use_norm |= corner.normal.is_some();
            use_uv |= corner.uv.is_some();
            let existing = verts.get(&corner).map(|f| f.index);
            existing.unwrap_or_else(|| {
                // Lazy-eval insertion so the indices stay correct.
                verts.insert(corner);
                index += 1;
                index - 1
            })
        };

        for poly in &group.polys {
            if poly.0.len() != 3 {
                continue;
            }
            tris.push([
                dedup_corner(&poly.0[0]),
                dedup_corner(&poly.0[1]),
                dedup_corner(&poly.0[2]),
            ]);
        }

        let mut verts: Vec<_> = verts.into_iter().collect();
        verts.sort_by(|a, b| a.index.cmp(&b.index));

        let mut tmp = Self {
            bvh: None,
            tris,
            verts: verts
                .iter()
                .map(|f| object.data.position[f.pos].into())
                .collect(),
            normals: use_norm.then(|| {
                verts
                    .iter()
                    .map(|f| object.data.normal[f.normal.unwrap()].into())
                    .collect()
            }),
            vert_cols: None,
            vert_uv: use_uv.then(|| {
                verts
                    .iter()
                    .map(|f| object.data.texture[f.uv.unwrap()].into())
                    .collect()
            }),
        };
        // tmp.create_bvh();
        tmp
    }
}
