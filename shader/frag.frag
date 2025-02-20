#version 450

layout(location = 0) out vec4 outColor;
layout(binding = 0, rgba32f) uniform image2D img;
layout(push_constant, std430) uniform pc { uint frameCounter; };

void main() {
  ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
  outColor = imageLoad(img, pixelCoords) / frameCounter;
}
