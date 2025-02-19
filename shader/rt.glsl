#version 450 core

layout(binding = 0, rgba32f) uniform image2D img;

void main() {
  // Write a dummy value at 0, 0.
  imageStore(img, ivec2(0, 0), vec4(1.0, 0.0, 0.0, 1.0));
}
