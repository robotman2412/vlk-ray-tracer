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
};

/* ==== RAY INTERSECTION TESTS ==== */

HitInfo rayTestSphere(Ray ray, uint obj) {
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

  return hit;
}

HitInfo rayTestPlane(Ray ray, uint obj) {
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
  vec3 pos = ray.pos + ray.pos * hit.dist;
  if (abs(pos.x) > 1 || abs(pos.y) > 1) {
    hit.dist = 1.0 / 0.0;
    return hit;
  }

  hit.pos = (objects[obj].transform.matrix * vec4(ray.pos, 1)).xyz;
  hit.normal = normalize(
      (objects[obj].transform.matrix * vec4(0, 0, sign(ray.pos.z), 1)).xyz);
  hit.isEntry = true;

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

void main() {
  ivec2 imgSize = imageSize(img);
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  if (pixelCoords.x >= imgSize.x || pixelCoords.y >= imgSize.y) {
    return;
  }
  vec4 prevColor = frameCounter > 1 ? imageLoad(img, pixelCoords) : vec4(0.0);

  float dist = float(imgSize.x) * 0.5 / camVFov;
  Ray ray;
  ray.pos = (camMatrix * vec4(0, 0, 0, 1)).xyz;
  ray.normal = normalize(
      (camMatrix * (vec4(pixelCoords, dist, 0) - 0.5 * vec4(imgSize, 0, 0)))
          .xyz);

  vec4 color;

  HitInfo bestHit = rayTest(ray);
  if (isinf(bestHit.dist)) {
    color = skybox.skyColor;
  } else {
    color = vec4(objects[bestHit.obj].transform.matrix[0]);
  }

  imageStore(img, pixelCoords, prevColor + color);
}
