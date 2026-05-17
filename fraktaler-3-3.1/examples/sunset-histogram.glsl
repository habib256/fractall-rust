// histogram colouring with SunsetColors palette
// <https://mathematica.stackexchange.com/a/30509>

uniform float power;

const vec3 sunsetColors[7] = vec3[7]
( vec3(0, 0, 0)
, vec3(0.372793, 0.1358, 0.506503)
, vec3(0.788287, 0.259816, 0.270778)
, vec3(0.979377, 0.451467, 0.0511329)
, vec3(1., 0.682688, 0.129771)
, vec3(1., 0.882236, 0.491094)
, vec3(1, 1, 1)
);

vec3 sunset(float x)
{
  x = 7.0 * clamp(x, 0.0, 1.0);
  int i = int(floor(x));
  float t = x - floor(x);
  return mix
    ( sunsetColors[clamp(i, 0, 6)]
    , sunsetColors[clamp(i + 1, 0, 6)]
    , t
    );
}

vec3 colour(void)
{
  float h = getHistogram(float(getN0()) + getNF());
  h = pow(h, power);
  return sunset(h);
}
