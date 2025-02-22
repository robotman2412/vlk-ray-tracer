#version 450

layout(location = 0) out vec4 outColor;
layout(binding = 0, rgba32f) uniform image2D img;
layout(push_constant, std430) uniform pc { uint frameCounter; };

vec3 aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
  vec4 baseColor = imageLoad(img, pixelCoords) / frameCounter;
  // outColor = baseColor / (baseColor + vec4(1));
  // outColor = log(baseColor + vec4(1));
  outColor = vec4(aces(baseColor.xyz), 0);
}
