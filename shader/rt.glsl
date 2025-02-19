#version 450 core

// Ray-tracing is clustered in 8x8 pixel tiles.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform image2D img;

void main() {
  // Write the sub-tile coords to the output pixel.
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  vec4 color = vec4(pixelCoords.x * 0.125, pixelCoords.y * 0.125, 0.0, 1.0);
  imageStore(img, pixelCoords, color);
}
