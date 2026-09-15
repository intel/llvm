// RUN: %clang_cc1 -triple spir64 -fdeclare-spirv-builtins -fconvergent-functions \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// -fconvergent-functions gives the declarations a convergent to remove.
typedef float float4 __attribute__((ext_vector_type(4)));

// Attr.Const: speculatable, not convergent.
float use_dot(float4 v1, float4 v2) { return __spirv_Dot(v1, v2); }

// Not Attr.Const: unchanged.
void use_barrier() { __spirv_ControlBarrier(2, 2, 912); }

// CHECK: declare {{.*}}@_Z11__spirv_DotDv4_fS_{{.*}} #[[DOT:[0-9]+]]
// CHECK: declare {{.*}}@_Z22__spirv_ControlBarrieriii{{.*}} #[[BAR:[0-9]+]]

// CHECK: attributes #[[DOT]] = {
// CHECK-NOT: convergent
// CHECK-SAME: speculatable
// CHECK-SAME: }

// CHECK: attributes #[[BAR]] = {
// CHECK-SAME: convergent
// CHECK-NOT: speculatable
// CHECK-SAME: }
