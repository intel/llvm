// Note: This isn't really target specific and should be switched to spir when
// it's enabled for it.

// REQUIRES: cuda
// RUN: %clang -fsycl -fsyntax-only -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s

// Check that mixed calls with long double don't cause compile errors on the
// device. Long double is not supported on the device for cmath built-ins. We
// can't make this an error during overload resolution, because in SYCL all
// functions are semantically checked for both host and device.

#include <cmath>

long double f(double f, double d, long double ld, int *pi) {
  long double r = 0.l;
  r = std::fmod(d, ld);
  r = std::remquo(d, ld, pi);
  r = std::fma(f, d, ld);
  return r;
}
