// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -fdeclare-spirv-builtins -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK: call {{.*}}@_Z11__spirv_DotDv4_fS_({{.*}}) #[[CALL:[0-9]+]]
// CHECK: declare {{.*}}@_Z11__spirv_DotDv4_fS_({{.*}}) #[[DOT:[0-9]+]]
float use_dot(float4 a, float4 b) { return __spirv_Dot(a, b); }
// CHECK: declare {{.*}}@_Z17__spirv_ocl_u_minjj({{.*}}) #[[DOT]]
unsigned use_min(unsigned x, unsigned y) { return __spirv_ocl_u_min(x, y); }

// Const, but not speculatable in the builtin table.
// CHECK: declare {{.*}}@_Z30__spirv_GenericPtrMemSemanticsPU3AS4Ki({{.*}}) #[[SEM:[0-9]+]]
int use_sem(const __generic int *p) { return __spirv_GenericPtrMemSemantics(p); }

// An explicitly 'convergent' redeclaration is not made 'speculatable'.
// CHECK: declare {{.*}}@_Z17__spirv_ocl_u_maxjj({{.*}}) #[[SEM]]
unsigned __attribute__((overloadable, convergent)) __spirv_ocl_u_max(unsigned x, unsigned y);
unsigned use_max(unsigned x, unsigned y) { return __spirv_ocl_u_max(x, y); }

// Definitions are not 'speculatable', whether they come before or after a use.
// CHECK: define {{.*}}@_Z17__spirv_ocl_u_absj({{.*}}) #[[DEF:[0-9]+]]
unsigned __attribute__((overloadable)) __spirv_ocl_u_abs(unsigned x) { return x; }
unsigned use_abs(int x) { return __spirv_ocl_s_abs(x); }
// CHECK: define {{.*}}@_Z17__spirv_ocl_s_absi({{.*}}) #[[DEF]]
unsigned __attribute__((overloadable)) __spirv_ocl_s_abs(int x) { return x < 0 ? -x : x; }

// CHECK: attributes #[[DOT]] = {
// CHECK-NOT: convergent
// CHECK-SAME: speculatable
// CHECK: attributes #[[SEM]] = {
// CHECK-SAME: convergent
// CHECK-NOT: speculatable
// CHECK-SAME: }
// CHECK: attributes #[[DEF]] = {
// CHECK-SAME: convergent
// CHECK-NOT: speculatable
// CHECK-SAME: }
// CHECK: attributes #[[CALL]] = {
// CHECK-NOT: speculatable
// CHECK-SAME: }
