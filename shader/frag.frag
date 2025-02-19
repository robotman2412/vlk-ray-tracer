#version 450

layout(location = 0) out vec4 outColor;
layout(binding = 0, rgba32f) uniform image2D img;

void main() {
  // Set out color based on position for debugging.
  ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
  outColor = imageLoad(img, pixelCoords);
}
