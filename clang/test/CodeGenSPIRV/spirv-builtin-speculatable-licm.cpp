// RUN: %clang_cc1 -triple spir64 -fdeclare-spirv-builtins -fconvergent-functions \
// RUN:   -O1 -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | opt -passes='sroa,loop-simplify,loop-mssa(licm)' -S | FileCheck %s

// licm hoists the invariant call out of the conditional block. The pass list is
// explicit so only the position of the call can differ; -O0 would be optnone.
typedef float float2 __attribute__((ext_vector_type(2)));

float f(float2 a, float2 b, const float *p, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i)
    if (p[i] > 0.f)
      acc += __spirv_Dot(a, b);
  return acc;
}

// CHECK-LABEL: define {{.*}}@_Z1fDv2_fS_PKfi
// CHECK: entry:
// CHECK-NEXT: %call = call spir_func float @_Z11__spirv_DotDv2_fS_
// CHECK: if.then:
// CHECK-NOT: call spir_func {{.*}}@_Z11__spirv_DotDv2_fS_
// CHECK: if.end:
