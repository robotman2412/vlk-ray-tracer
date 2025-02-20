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
