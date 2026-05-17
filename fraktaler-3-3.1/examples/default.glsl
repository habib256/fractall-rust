uniform float brightness;
uniform float contrast;
uniform float exposure;
uniform float gamma;

vec3 colour(void)
{
  vec2 de = getDE();
  float v = 0.875 + 0.03125 * log2(dot(de, de));
  v += brightness;
  v -= 0.5;
  v *= exp2(contrast);
  v += 0.5;
  v = clamp(v, 0.0, 1.0);
  v = pow(v, 1.0 / gamma);
  v *= exp2(exposure);
  return vec3(v);
}
