// colouring based on emndl with various enhancements
// perceptual hue adjustment based on code from mrob

uniform float speed;
uniform float offset;
uniform float brightness;
uniform float contrast;
uniform float gamma;
uniform float exposure;
uniform vec3 interior;

float lin_interp(float x, float domain_low, float domain_hi, float range_low, float range_hi)
{
  if ((x >= domain_low) && (x <= domain_hi))
  {
    x = (x - domain_low) / (domain_hi - domain_low);
    x = range_low + x * (range_hi - range_low);
  }
  return x;
}

float pvp_adjust_3(float x)
{
  // red
  x = lin_interp(x, 0.00, 0.125, -0.050, 0.090);
  // orange
  x = lin_interp(x, 0.125, 0.25,  0.090, 0.167);
  // yellow
  x = lin_interp(x, 0.25, 0.375,  0.167, 0.253);
  // chartreuse
  x = lin_interp(x, 0.375, 0.50,  0.253, 0.383);
  // green
  x = lin_interp(x, 0.50, 0.625,  0.383, 0.500);
  // teal
  x = lin_interp(x, 0.625, 0.75,  0.500, 0.667);
  // blue
  x = lin_interp(x, 0.75, 0.875,  0.667, 0.800);
  // purple
  x = lin_interp(x, 0.875, 1.00,  0.800, 0.950);
  return x;
}

vec3 colour(void)
{
  // FIXME hue loses precision at higher iteration counts
  float h = (float(getN0()) + getNF()) / exp2(speed);
  float s = getT() > 0.0 ? 0.4 : 0.6;
  float v = 0.25 + 0.25 * log2(length(getDE()));
  h -= floor(h);
  h = pvp_adjust_3(h);
  v += offset;
  v = clamp(v, 0.0, 1.0);
  vec3 rgb = hsv2rgb(vec3(h, s, v));
  if (length(getDE()) <= 0.0)
  {
    rgb = interior;
  }
  rgb += vec3(brightness);
  rgb += vec3(0.5);
  rgb *= exp2(contrast);
  rgb -= vec3(0.5);
  rgb = clamp(rgb, vec3(0.0), vec3(1.0));
  rgb = pow(rgb, vec3(1.0 / gamma));
  rgb *= exp2(exposure);
  return rgb;
}
