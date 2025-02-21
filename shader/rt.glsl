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
  uint _padding0;
  uint _padding1;
  uint _padding2;
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

struct HitInfo {
  vec3 pos;
  vec3 normal;
  uint obj;
  float dist;
  bool isEntry;
};

/* ==== LAYOUT DEFINITIONS ==== */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform image2D img;
layout(binding = 1, std430) buffer ObjectBuffer { Object objects[]; };
layout(binding = 2, std430) buffer SkyboxBuffer { Skybox skybox; };
layout(push_constant, std430) uniform ParamPC {
  mat4 camMatrix;
  float camVFov;
  uint frameCounter;
  uint objectCount;
  uint maxReflect;
  uint maxRefract;
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

  if (b < 0.0) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }

  if (b < 0.00000001) {
    if (a > 0.00000001) {
      hit.dist = a;
    } else {
      hit.dist = 1.0 / 0.0;
      return hit;
    }
  } else {
    float dist0 = a + sqrt(b);
    float dist1 = a - sqrt(b);
    if (dist0 < dist1 && dist0 > 0.00000001) {
      hit.dist = dist0;
    } else if (dist1 > 0.00000001) {
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
  hit.dist = -ray.pos.z / ray.normal.z;

  if (abs(ray.normal.z) < 0.00000001 || hit.dist < 0.00000001) {
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

HitInfo rayTestObject(Ray ray, uint obj) {
  switch (objects[obj].type) {
  case 0:
    return rayTestSphere(ray, obj);
  case 1:
    return rayTestPlane(ray, obj);
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

vec4 rayTrace(Ray ray, uint seed) {
  uint reflectLeft = maxReflect;
  uint refractLeft = maxRefract;

  // Mix up the RNG a tiny bit.
  uint rngState = seed;
  splitmix32(rngState);
  splitmix32(rngState);

  vec4 colMask = vec4(1);
  vec4 color = vec4(0);

  while (true) {
    HitInfo hit = rayTest(ray);

    if (!isinf(hit.dist)) {
      color += colMask * objects[hit.obj].physProp.emission;
      colMask *= objects[hit.obj].physProp.color;

      if (!hit.isEntry ||
          randFloat(rngState) >= objects[hit.obj].physProp.opacity) {
        // Do refraction.
        if (refractLeft == 0)
          return color;
        refractLeft--;

        // Get normal and IORs.
        float ratio;
        vec3 normal;
        if (hit.isEntry) {
          ratio = 1.0 / objects[hit.obj].physProp.ior;
          normal = -hit.normal;
        } else {
          ratio = objects[hit.obj].physProp.ior;
          normal = hit.normal;
        }

        // Determine refraction angle.
        float inDot = clamp(dot(-ray.normal, normal), -1, 1);
        float det = max(0, 1 - ratio * ratio * (1 - inDot * inDot));
        ray.pos = hit.pos;
        ray.normal = ray.normal * ratio + normal * (sqrt(det) - ratio * inDot);
        ray.normal = normalize(ray.normal);

      } else {
        // Do reflection.
        if (reflectLeft == 0)
          return color;
        reflectLeft--;

        vec3 diffNormal =
            normalize(randHemisphereVec(rngState, hit.normal) + hit.normal);
        vec3 specNormal = normalize(
            ray.normal - 2 * dot(ray.normal, hit.normal) * hit.normal);

        ray.pos = hit.pos;
        ray.normal = specNormal + (diffNormal - specNormal) *
                                      objects[hit.obj].physProp.roughness;
        ray.normal = normalize(ray.normal);
      }
    } else {
      // No hit; sample skybox color.
      float coeff = clamp(ray.normal.y * 3, -1, 1);
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
}

void main() {
  ivec2 imgSize = imageSize(img);
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  if (pixelCoords.x >= imgSize.x || pixelCoords.y >= imgSize.y) {
    return;
  }
  vec4 prevColor = frameCounter > 1 ? imageLoad(img, pixelCoords) : vec4(0.0);

  float dist = float(imgSize.y) * 0.5 / camVFov;
  Ray ray;
  ray.pos = (camMatrix * vec4(0, 0, 0, 1)).xyz;
  ray.normal = normalize(
      (camMatrix * (vec4(pixelCoords, dist, 0) - 0.5 * vec4(imgSize, 0, 0)))
          .xyz);

  // Create primitive RNG seed.
  uint seed = frameCounter * (1 + pixelCoords.x + pixelCoords.y * imgSize.x);

  vec4 color = vec4(rayTrace(ray, seed));

  // The fourth channel is for debug info and is not accumulated.
  prevColor.w = 0;
  imageStore(img, pixelCoords, prevColor + color);
}
