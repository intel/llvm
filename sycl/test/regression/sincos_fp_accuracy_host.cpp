// Verifies that sycl::sincos does not crash the compiler during host
// compilation with -ffp-accuracy=high. The SYCL sincos has a different
// signature (float sincos(float, float*)) from the GNU sincos 
// (void sincos(float, float*, float*)) and must not be replaced with the
// fpbuiltin_sincos intrinsic.

// RUN: %clangxx -fsycl -ffp-accuracy=high -fno-math-errno %s -S -emit-llvm \
// RUN: -o - | FileCheck %s
#include <sycl/sycl.hpp>

// CHECK-NOT: call void @llvm.fpbuiltin.sincos
// CHECK: call {{.*}}float @_ZN4sycl{{.*}}sincos_impl{{.*}}(float {{.*}}, ptr {{.*}})

void test(sycl::queue &q) {
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class SinCosKernel>(sycl::range<1>(4), [=](sycl::id<1> idx) {
      float x = 1.0f;
      float c;
      float s = sycl::sincos(x, &c);
    });
  });
}

