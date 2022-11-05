// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unkown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NO-INLINE
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unkown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-INLINE

#include "sycl.hpp"

int main() {
  sycl::queue q;

  // CHECK: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E10KernelName()
  //
  // CHECK-NO-INLINE: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
  // CHECK-INLINE-NOT: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv
  q.submit([&](sycl::handler &h) { h.parallel_for<class KernelName>([] {}); });


  // CHECK: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E16KernelNameInline()
  // CHECK-NOT: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_ENKUlvE_clEv
  q.submit([&](sycl::handler &h) { h.parallel_for<class KernelNameInline>([]() __attribute__((always_inline)) {}); });

  // CHECK: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_E18KernelNameNoInline()
  // CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_ENKUlvE_clEv
  q.submit([&](sycl::handler &h) { h.parallel_for<class KernelNameNoInline>([]() __attribute__((noinline)) {}); });

  /// The flag is ignored for ESIMD kernels
  // CHECK: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE2_clES2_E15KernelNameESIMD()
  // CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE2_clES2_ENKUlvE_clEv
  q.submit([&](sycl::handler &h) { h.parallel_for<class KernelNameESIMD>([]() __attribute__((sycl_explicit_simd)) {}); });

  return 0;
}
