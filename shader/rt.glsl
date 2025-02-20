#version 450 core

/* ==== TYPE DEFINITIONS ==== */

struct Transform {
  mat4 matrix;
  mat4 invMatrix;
};

struct PhysProp {
  float ior;
  float opacity;
  float roughness;
  vec4 color;
  vec4 emission;
};

struct Object {
  Transform transform;
  PhysProp physProp;
  uint type;
};

struct Skybox {
  float sunRadius;
  vec4 groundColor;
  vec4 horizonColor;
  vec4 skyboxColor;
  vec4 sunColor;
  vec4 sunDirection;
};

struct Ray {
  vec3 pos;
  vec3 normal;
};

struct HitInfo {
  vec3 pos;
  vec3 normal;
  float dist;
  bool isEntry;
};

/* ==== LAYOUT DEFINITIONS ==== */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform image2D img;
layout(binding = 1, std430) buffer objectBuffer { Object objects[]; };
layout(push_constant, std430) uniform ParamPC {
  mat4 camMatrix;
  float camVFov;
  uint frameCounter;
  Skybox skybox;
};

/* ==== RAY INTERSECTION TESTS ==== */

HitInfo rayTestCircle(Ray ray, int obj) {
  ray.pos = (objects[obj].transform.invMatrix * vec4(ray.pos, 1)).xyz;
  ray.normal =
      normalize((objects[obj].transform.invMatrix * vec4(ray.normal, 0)).xyz);

  float a = -dot(ray.normal, ray.pos);
  float raySqrMag = dot(ray.pos, ray.pos);
  float b = a * a - raySqrMag + 1;
  HitInfo hit;

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

HitInfo rayTestPlane(Ray ray, int obj) {
  ray.pos = (objects[obj].transform.invMatrix * vec4(ray.pos, 1)).xyz;
  ray.normal =
      normalize((objects[obj].transform.invMatrix * vec4(ray.normal, 0)).xyz);

  HitInfo hit;
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

HitInfo rayTest(Ray ray, int obj) {
  switch (objects[obj].type) {
  case 0:
    return rayTestCircle(ray, obj);
  case 1:
    return rayTestPlane(ray, obj);
  }
}

/* ==== MAIN SHADER CODE ==== */

void main() {
  ivec2 imgSize = imageSize(img);
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  if (pixelCoords.x >= imgSize.x || pixelCoords.y >= imgSize.y) {
    return;
  }
  vec4 prevColor = frameCounter > 1 ? imageLoad(img, pixelCoords) : vec4(0.0);

  float fWidth = float(imgSize.x);
  vec4 color =
      vec4(pixelCoords.x % 64 / 64.0, pixelCoords.y % 64 / 64.0,
           pixelCoords.x / fWidth, camVFov + objects[0].physProp.color.x);
  imageStore(img, pixelCoords, prevColor + color);
}
