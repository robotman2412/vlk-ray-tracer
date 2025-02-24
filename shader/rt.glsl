#version 450 core

/* ==== BUFFER DEFINITIONS ==== */

struct Transform {
  mat4 matrix;
  mat4 invMatrix;
};

struct PhysProp {
  float ior;
  float opacity;
  float roughness;
  uint _padding0;
  vec4 color;
  vec4 emission;
};

struct Object {
  Transform transform;
  PhysProp physProp;
  uint type;
  uint mesh;
  uint _padding1;
  uint _padding2;
};

struct Mesh {
  uint numTris;
  uint numVerts;
  uint triOffset;
  uint vertOffset;
  uint normOffset;
  uint vcolOffset;
  uint uvOffset;
};

struct Skybox {
  vec4 groundColor;
  vec4 horizonColor;
  vec4 skyColor;
  vec4 sunColor;
  vec4 sunDirection;
  float sunRadius;
};

/* ==== TYPE DEFINITIONS ==== */

struct Ray {
  vec3 pos;
  vec3 normal;
};

struct TriHitInfo {
  // Global triangle number
  uint tri;
  // Barycentric co-ordinates.
  float u, v;
  // Hit distance from ray origin.
  float dist;
};

struct HitInfo {
  vec3 pos;
  float dist;
  vec3 normal;
  uint obj;
  PhysProp physProp;
  bool isEntry;
};

/* ==== LAYOUT DEFINITIONS ==== */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform image2D img;
layout(binding = 1, std430) buffer SkyboxBuffer { Skybox skybox; };
layout(binding = 2, std430) buffer ObjectBuffer { Object objects[]; };
layout(binding = 3, std430) buffer MeshBuffer { Mesh meshes[]; };
layout(binding = 4, std430) buffer TriBuffer { uint tris[]; };
layout(binding = 5, std430) buffer VertBuffer { vec3 verts[]; };
layout(binding = 6, std430) buffer NormBuffer { vec3 norms[]; };
layout(binding = 7, std430) buffer VcolBuffer { vec4 vcols[]; };
// layout(binding = 8, std430) buffer UvBuffer { vec2 uvs[]; };
layout(push_constant, std430) uniform ParamPC {
  mat4 camMatrix;
  float camVFov;
  uint frameCounter;
  uint maxBounce;
  uint objectCount;
};

/* ==== RAY INTERSECTION TESTS ==== */

HitInfo rayTestSphere(Ray ray, uint obj) {
  Ray globalRay = ray;
  ray.pos = (objects[obj].transform.invMatrix * vec4(ray.pos, 1)).xyz;
  ray.normal =
      normalize((objects[obj].transform.invMatrix * vec4(ray.normal, 0)).xyz);

  float a = -dot(ray.normal, ray.pos);
  float raySqrMag = dot(ray.pos, ray.pos);
  float b = a * a - raySqrMag + 1;
  HitInfo hit;
  hit.obj = obj;
  hit.physProp = objects[obj].physProp;

  if (b < 0.0) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }

  if (b < 0.00001) {
    if (a > 0.000001) {
      hit.dist = a;
    } else {
      hit.dist = 1.0 / 0.0;
      return hit;
    }
  } else {
    float dist0 = a + sqrt(b);
    float dist1 = a - sqrt(b);
    if ((dist1 <= 0.00001 || dist0 < dist1) && dist0 > 0.00001) {
      hit.dist = dist0;
    } else if (dist1 > 0.00001) {
      hit.dist = dist1;
    } else {
      hit.dist = 1.0 / 0.0;
      return hit;
    }
  }

  vec3 pos = ray.pos + ray.normal * hit.dist;
  hit.pos = (objects[obj].transform.matrix * vec4(pos, 1)).xyz;
  hit.normal = normalize((objects[obj].transform.matrix * vec4(pos, 0)).xyz);
  hit.isEntry = raySqrMag > 1;
  hit.dist = length(globalRay.pos - hit.pos);

  return hit;
}

HitInfo rayTestPlane(Ray ray, uint obj) {
  Ray globalRay = ray;
  ray.pos = (objects[obj].transform.invMatrix * vec4(ray.pos, 1)).xyz;
  ray.normal =
      normalize((objects[obj].transform.invMatrix * vec4(ray.normal, 0)).xyz);

  HitInfo hit;
  hit.obj = obj;
  hit.physProp = objects[obj].physProp;
  hit.dist = -ray.pos.z / ray.normal.z;

  if (abs(ray.normal.z) < 0.00001 || hit.dist < 0.00001) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }
  vec3 pos = ray.pos + ray.normal * hit.dist;
  if (abs(pos.x) > 1 || abs(pos.y) > 1) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }

  hit.pos = (objects[obj].transform.matrix * vec4(pos, 1)).xyz;
  hit.normal = normalize(
      (objects[obj].transform.matrix * vec4(0, 0, sign(ray.pos.z), 1)).xyz);
  hit.isEntry = true;
  hit.dist = length(globalRay.pos - hit.pos);

  return hit;
}

TriHitInfo rayTestTri(Ray ray, uint tri) {
  TriHitInfo hit;
  hit.tri = tri;
  hit.dist = 1.0 / 0.0;

  vec3 a = verts[tris[tri]];
  vec3 b = verts[tris[tri + 1]];
  vec3 c = verts[tris[tri + 2]];

  vec3 ab = b - a;
  vec3 ac = c - a;
  vec3 pvec = cross(ray.normal, ac);
  float det = dot(ab, pvec);

  if (abs(det) < 0.00001) {
    return hit;
  }

  float invDet = 1 / det;
  vec3 tvec = ray.pos - a;
  float u = invDet * dot(tvec, pvec);
  if (u < 0 || u > 1) {
    return hit;
  }

  vec3 qvec = cross(tvec, ab);
  float v = invDet * dot(ray.normal, qvec);
  if (v < 0 || u + v > 1) {
    return hit;
  }

  hit.dist = invDet * dot(ac, qvec);
  if (hit.dist > 0.0001) {
    hit.u = u;
    hit.v = v;
    return hit;
  } else {
    hit.dist = 1.0 / 0.0;
    return hit;
  }
}

HitInfo rayTestMesh(Ray ray, uint obj) {
  Ray globalRay = ray;
  ray.pos = (objects[obj].transform.invMatrix * vec4(ray.pos, 1)).xyz;
  ray.normal =
      normalize((objects[obj].transform.invMatrix * vec4(ray.normal, 0)).xyz);

  Mesh mesh = meshes[objects[obj].mesh];

  TriHitInfo bestHit;
  bestHit.dist = 1.0 / 0.0;
  for (uint i = 0; i < mesh.numTris; i++) {
    TriHitInfo hit = rayTestTri(ray, mesh.triOffset + i * 3);
    if (hit.dist < bestHit.dist) {
      bestHit = hit;
    }
  }

  HitInfo hit;
  if (isinf(bestHit.dist)) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }
  hit.pos = ray.pos + hit.dist * ray.normal;
  hit.physProp = objects[obj].physProp;

  uint a = tris[bestHit.tri];
  uint b = tris[bestHit.tri + 1];
  uint c = tris[bestHit.tri + 2];

  if (mesh.normOffset == uint(-1)) {
    hit.normal = normalize(cross(verts[c] - verts[a], verts[b] - verts[a]));
  } else {
    hit.normal = (1 - bestHit.u - bestHit.v) * norms[a];
    hit.normal += bestHit.u * norms[b];
    hit.normal += bestHit.v * norms[c];
    // Normalization happens later; doing it here is redundant.
  }
  hit.isEntry = dot(ray.normal, hit.normal) > 0;

  if (mesh.vcolOffset != uint(-1)) {
    vec4 vcol;
    vcol = (1 - bestHit.u - bestHit.v) * vcols[a];
    vcol += bestHit.u * vcols[b];
    vcol += bestHit.v * vcols[c];
    hit.physProp.color *= vcol;
  }

  hit.pos = (objects[obj].transform.matrix * vec4(hit.pos, 1)).xyz;
  hit.normal = (objects[obj].transform.matrix * vec4(hit.normal, 0)).xyz;
  hit.normal = normalize(hit.normal);
  hit.dist = length(globalRay.pos - hit.pos);
  return hit;
}

HitInfo rayTestObject(Ray ray, uint obj) {
  switch (objects[obj].type) {
  case 0:
    return rayTestSphere(ray, obj);
  case 1:
    return rayTestPlane(ray, obj);
  case 2:
    return rayTestMesh(ray, obj);
  }
}

HitInfo rayTest(Ray ray) {
  HitInfo bestHit;
  bestHit.dist = 1.0 / 0.0;

  for (uint i = 0; i < objectCount; i++) {
    HitInfo hit = rayTestObject(ray, i);
    if (hit.dist < bestHit.dist) {
      bestHit = hit;
    }
  }

  return bestHit;
}

/* ==== MAIN SHADER CODE ==== */

// https://stackoverflow.com/a/52056161
uint splitmix32(inout uint state) {
  uint z = (state += 0x9e3779b9);
  z ^= z >> 16;
  z *= 0x21f0aaad;
  z ^= z >> 15;
  z *= 0x735a2d97;
  z ^= z >> 15;
  return z;
}

float randFloat(inout uint rngState) {
  return splitmix32(rngState) / float(uint(-1));
}

float randNormal(inout uint rngState) {
  float t = 6.283185307179586 * randFloat(rngState);
  float r = sqrt(log(randFloat(rngState)) * -2);
  return r * cos(t);
}

vec3 randUnitVec(inout uint rngState) {
  return normalize(
      vec3(randNormal(rngState), randNormal(rngState), randNormal(rngState)));
}

vec3 randHemisphereVec(inout uint rngState, vec3 relativeTo) {
  vec3 tmp = randUnitVec(rngState);
  if (dot(tmp, relativeTo) < 0) {
    return -tmp;
  } else {
    return tmp;
  }
}

vec4 rayTrace(Ray ray, inout uint rngState) {
  uint bounceLeft = maxBounce;

  vec4 colMask = vec4(1);
  vec4 color = vec4(0);

  while (bounceLeft > 0 && colMask.x + colMask.y + colMask.z > 0.001) {
    HitInfo hit = rayTest(ray);
    bounceLeft--;

    if (!isinf(hit.dist)) {
      color += colMask * objects[hit.obj].physProp.emission;
      colMask *= objects[hit.obj].physProp.color;
      bool doReflect = true;
      vec3 normal = hit.isEntry ? hit.normal : -hit.normal;

      if (randFloat(rngState) >= objects[hit.obj].physProp.opacity) {
        // Get normal and IORs.
        float ratio;
        if (hit.isEntry) {
          ratio = 1.0 / objects[hit.obj].physProp.ior;
        } else {
          ratio = objects[hit.obj].physProp.ior;
        }

        // Determine refraction angle.
        float inDot = clamp(-dot(ray.normal, normal), -1, 1);
        float det = 1 - ratio * ratio * (1 - inDot * inDot);
        if (det >= 0) {
          ray.pos = hit.pos;
          ray.normal =
              ray.normal * ratio + normal * (ratio * inDot - sqrt(det));
          ray.normal = normalize(ray.normal);
          doReflect = false;
        }
      }
      if (doReflect) {
        // Do reflection.
        vec3 diffNormal = normalize(randUnitVec(rngState) + normal);
        vec3 specNormal =
            normalize(ray.normal - 2 * dot(ray.normal, hit.normal) * normal);

        ray.pos = hit.pos;
        ray.normal = specNormal + (diffNormal - specNormal) *
                                      objects[hit.obj].physProp.roughness;
        ray.normal = normalize(ray.normal);
      }
    } else {
      // No hit; sample skybox color.
      float coeff = clamp(ray.normal.y * 4, -1, 1);
      vec4 base;
      if (coeff >= 0) {
        base = skybox.horizonColor +
               (skybox.groundColor - skybox.horizonColor) * coeff;
      } else {
        base = skybox.horizonColor +
               (skybox.skyColor - skybox.horizonColor) * -coeff;
      }

      float sunDot = dot(ray.normal, skybox.sunDirection.xyz);
      if (sunDot >= skybox.sunRadius) {
        float sunCoeff = (sunDot - skybox.sunRadius) / (1.0 - skybox.sunRadius);
        color += colMask * (base + (skybox.sunColor - base) * sunCoeff);
      } else {
        color += colMask * base;
      }

      return color;
    }
  }

  return color;
}

void main() {
  ivec2 imgSize = imageSize(img);
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  if (pixelCoords.x >= imgSize.x || pixelCoords.y >= imgSize.y) {
    return;
  }
  vec4 prevColor = frameCounter > 1 ? imageLoad(img, pixelCoords) : vec4(0.0);

  // Create primitive RNG seed.
  uint rngState =
      frameCounter * (1 + pixelCoords.x + pixelCoords.y * imgSize.x);
  splitmix32(rngState);
  splitmix32(rngState);

  float dist = float(imgSize.y) * 0.5 / camVFov;
  Ray ray;
  ray.pos = (camMatrix * vec4(0, 0, 0, 1)).xyz;
  vec2 randOff = vec2(randFloat(rngState), randFloat(rngState)) - 0.5;
  vec2 pixelCoordsf = vec2(pixelCoords) + randOff;
  ray.normal = normalize(
      (camMatrix * (vec4(pixelCoordsf, dist, 0) - 0.5 * vec4(imgSize, 0, 0)))
          .xyz);

  vec4 color = vec4(rayTrace(ray, rngState));

  // The fourth channel is for debug info and is not accumulated.
  prevColor.w = 0;
  imageStore(img, pixelCoords, prevColor + color);
}
