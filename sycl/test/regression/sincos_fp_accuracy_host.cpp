// Verifies that sycl::sincos does not crash the compiler during host
// compilation with -ffp-accuracy=high. The SYCL sincos has a different
// signature (float sincos(float, float*)) from the GNU sincos
// (void sincos(float, float*, float*)) and must not be replaced with the
// fpbuiltin_sincos intrinsic.

// RUN: %clangxx -fsycl -ffp-accuracy=high -fno-math-errno %s -S -emit-llvm \
// RUN: -o - | FileCheck %s
#include <sycl/builtins.hpp>

// CHECK-NOT: call {{.*}}@llvm.fpbuiltin.sincos
// CHECK: call {{.*}}float {{.*}}sycl{{.*}}sincos_impl{{.*}}(float {{.*}}, ptr {{.*}})
float host_sincos_sycl(float x, float *c) {
  float s = sycl::sincos(x, c);
  return s;
}
