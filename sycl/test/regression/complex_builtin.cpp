// RUN: %clangxx -fsycl-device-only %s -Werror=implicit-float-conversion -fsyntax-only
// REQUIRES: linux

#include <cmath>
#include <sycl/detail/core.hpp>

SYCL_EXTERNAL
double __complex__ run_divdc3(double a, double b, double c, double d) {
  double __complex__ ret = __divdc3(a, b, c, d);
  return ret;
}

SYCL_EXTERNAL
double __complex__ run_muldc3(double a, double b, double c, double d) {
  double __complex__ ret = __muldc3(a, b, c, d);
  return ret;
}

SYCL_EXTERNAL
float __complex__ run_cpowf(float __complex__ x, float __complex__ y) {
  float __complex__ ret = cpowf(x, y);
  return ret;
}

SYCL_EXTERNAL
double __complex__ run_cpow(double __complex__ x, double __complex__ y) {
  double __complex__ ret = cpow(x, y);
  return ret;
}
