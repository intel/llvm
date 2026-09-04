// RUN: %clangxx -fsycl -fsycl-device-only -ffp-accuracy=high -fno-math-errno \
// RUN: %s -S -emit-llvm -o - | FileCheck %s

#include <sycl/detail/defines_elementary.hpp>
// C++ sincos (not extern "C", so not a recognized builtin)
SYCL_EXTERNAL void sincos(double, double *, double *);

SYCL_EXTERNAL
void run_sincos(double x) {
  double s, c;
  // CHECK: call void @llvm.fpbuiltin.sincos.f64.p4.p4(double {{.*}}, ptr {{.*}}, ptr {{.*}})
  sincos(x, &s, &c);
}
