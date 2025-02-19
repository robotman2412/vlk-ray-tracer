#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
  // Set out color based on position for debugging.
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);
  x %= 256;
  y %= 256;
  outColor.x = float(x);
  outColor.x /= 255;
  outColor.y = float(y);
  outColor.y /= 255;
  outColor.z = 0.0;
  outColor.w = 0.0;
}
