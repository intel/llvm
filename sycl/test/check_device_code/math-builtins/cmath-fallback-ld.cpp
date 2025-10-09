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

// Check that unqualified calls to C-style math builtin dont't cause compile
// errors on the device. Unsuffixed calls should resolve to the double
// overloads, and not cause ambiguity. This is a regression test for a bug,
// where we used to define multiple overloads in the global namespace making the
// following code ambiguous. For example since we didn't define `long double
// fabs(long double)` but defined both `double fabs(double)` and `float
// fabs(float)` the call to `fabs(long double)` was ambiguous.

// NOTE: The fact that this compiles at all is already a (conformant) extension
//       by the C++ standard library implementation, because <cmath> is not
//       required (but allowed?) to put the <math.h> symbols into the global
//       namespace. We want to preserve this behavior for SYCL, to not break C++
//       code that relies on it.

long double g(long double ld, int i, int *pi, long l) {
  long double r = 0.l;
  r = fabs(ld);
  r = remquo(ld, ld, pi);
  r = fma(ld, ld, ld);
  r = fmax(ld, ld);
  r = frexp(ld, pi);
  r = ldexp(ld, i);
  r = scalbn(ld, i);
  r = scalbln(ld, l);
  r = ilogb(ld);
  return r;
}
