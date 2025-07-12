// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -x c++ %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that all LLVM-IR instructions that work with SPIR-V builtins
// are correctly translated into GenX counterparts (implemented in
// LowerESIMD.cpp)

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupLocalInvocationId(size_t *DoNotOptimize,
                                 uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_SubgroupLocalInvocationId();
  DoNotOptimize32[0] = __spirv_SubgroupLocalInvocationId() + 3;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupLocalInvocationId
  // CHECK: store i64 0, ptr addrspace(4) %DoNotOptimize, align 8
  // CHECK: store i32 3, ptr addrspace(4) %DoNotOptimize32, align 4
}

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupSize(size_t *DoNotOptimize, uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_SubgroupSize();
  DoNotOptimize32[0] = __spirv_SubgroupSize() + 7;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupSize
  // CHECK: store i64 1, ptr addrspace(4) %DoNotOptimize, align 8
  // CHECK: store i32 8, ptr addrspace(4) %DoNotOptimize32, align 4
}

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupMaxSize(size_t *DoNotOptimize, uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_SubgroupMaxSize();
  DoNotOptimize32[0] = __spirv_SubgroupMaxSize() + 9;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupMaxSize
  // CHECK: store i64 1, ptr addrspace(4) %DoNotOptimize, align 8
  // CHECK: store i32 10, ptr addrspace(4) %DoNotOptimize32, align 4
}
