// Blending test shader.

// If everything is configured correctly,
// the central part with checkerboard
// should be the same average brightness as
// the border region with mid-grey.

// Try changing scale between -2 and +2.
// Adjust number of samples in Quality dialog.

uniform float scale;

vec3 colour(void)
{
  vec2 c = getCoord() / vec2(getImageSize());
  if (isnan(c.x) || isnan(c.y))
  {
    return vec3(1,0,0);
  }
  else if (isinf(c.x) || isinf(c.y))
  {
    return vec3(0,0,1);
  }
  else if (0.25 < c.x && c.x < 0.75 && 0.25 < c.y && c.y < 0.75)
  {
    ivec2 k = ivec2(getCoord() * exp2(scale));
    int b = (k.x ^ k.y) & 1;
    return vec3(b);
  }
  else
  {
    return vec3(0.5);
  }
}
