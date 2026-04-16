// RUN: %clangxx -fsycl-device-only %s -Werror=implicit-float-conversion -fsyntax-only
// REQUIRES: linux

#include <sycl/detail/core.hpp>
#include <cmath>

SYCL_EXTERNAL
double __complex__ run_divdc3(double a, double b, double c, double d) {
  double __complex__ ret = __divdc3(a, b, c ,d);
  return ret;
}

SYCL_EXTERNAL
double __complex__ run_muldc3(double a, double b, double c, double d) {
  double __complex__ ret = __muldc3(a, b, c ,d);
  return ret;
}
