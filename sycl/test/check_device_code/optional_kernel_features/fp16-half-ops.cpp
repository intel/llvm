// RUN: %clangxx %s -fsycl -fsycl-device-only -O0 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O0
// RUN: %clangxx %s -fsycl -fsycl-device-only -O2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O2

// Check that the fp16 aspect is used by functions whose emitted code operates
// on 16-bit floating-point values, and not by code that only refers to
// sycl::half. The aspect is decided after optimization, so arithmetic that is
// optimized away does not use it.

#include <sycl/sycl.hpp>

// CHECK-LABEL: define {{.*}}@{{.*}}halfAddAssign
// CHECK-SAME:      !sycl_used_aspects ![[#FP16:]]
SYCL_EXTERNAL void halfAddAssign(sycl::half *in, sycl::half *out, size_t i) {
  out[i] += in[i];
}

// CHECK-LABEL: define {{.*}}@{{.*}}halfLess
// CHECK-SAME:      !sycl_used_aspects ![[#FP16]]
SYCL_EXTERNAL bool halfLess(sycl::half *in) { return in[0] < in[1]; }

// CHECK-LABEL: define {{.*}}@{{.*}}halfToFloat
// CHECK-SAME:      !sycl_used_aspects ![[#FP16]]
SYCL_EXTERNAL void halfToFloat(sycl::half *in, float *out) { *out = in[0]; }

// CHECK-LABEL: define {{.*}}@{{.*}}float16Mul
// CHECK-SAME:      !sycl_used_aspects ![[#FP16]]
SYCL_EXTERNAL void float16Mul(_Float16 *in, _Float16 *out) {
  *out = in[0] * in[1];
}

// CHECK-LABEL: define {{.*}}@{{.*}}halfDeadArith
// CHECK-O0-SAME:   !sycl_used_aspects ![[#FP16]]
// CHECK-O2-NOT:    !sycl_used_aspects
SYCL_EXTERNAL void halfDeadArith(sycl::half *in) {
  sycl::half h = in[0] + in[1];
  (void)h;
}

// CHECK-LABEL: define {{.*}}@{{.*}}halfCopyDeref
// CHECK-NOT:       !sycl_used_aspects
SYCL_EXTERNAL void halfCopyDeref(sycl::half *in, sycl::half *out) {
  *out = *in;
}

// CHECK-LABEL: define {{.*}}@{{.*}}halfCopyIndex
// CHECK-NOT:       !sycl_used_aspects
SYCL_EXTERNAL void halfCopyIndex(sycl::half *in, sycl::half *out, size_t i) {
  out[i] = in[i];
}

// CHECK-LABEL: define {{.*}}@{{.*}}halfPtrOnly
// CHECK-NOT:       !sycl_used_aspects
SYCL_EXTERNAL void halfPtrOnly(sycl::half *in, sycl::half **out) {
  *out = in + 1;
}

// CHECK: ![[#FP16]] = !{![[#ASPECT:]]}
// CHECK: ![[#ASPECT]] = !{!"fp16", i32 5}
