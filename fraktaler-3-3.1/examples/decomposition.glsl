uniform int count;

vec3 colour(void)
{
  int m = count;
  if (m <= 1)
  {
    m = 2;
  }
  float T = getT();
  T -= floor(T);
  T *= float(m);
  T = floor(T);
  T /= float(m - 1);
  return vec3(T);
}
