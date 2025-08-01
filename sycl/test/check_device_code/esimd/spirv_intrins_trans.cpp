// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -x c++ %s -o %t
// -O0 lowering, requires `-force-disable-esimd-opt` to disable all
// optimizations.
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -force-disable-esimd-opt -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that all LLVM-IR instructions that work with SPIR-V builtins
// are correctly translated into GenX counterparts (implemented in
// LowerESIMD.cpp)

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupLocalInvocationId(size_t *DoNotOptimize,
                                 uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_BuiltInSubgroupLocalInvocationId();
  DoNotOptimize32[0] = __spirv_BuiltInSubgroupLocalInvocationId() + 3;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupLocalInvocationId
  // CHECK: [[ZEXT0:%.*]] = zext i32 0 to i64
  // CHECK: store i64 [[ZEXT0]]
  // CHECK: add i32 0, 3
}

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupSize(size_t *DoNotOptimize, uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_BuiltInSubgroupSize();
  DoNotOptimize32[0] = __spirv_BuiltInSubgroupSize() + 7;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupSize
  // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
  // CHECK: store i64 [[ZEXT0]]
  // CHECK: add i32 1, 7
}

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void
kernel_SubgroupMaxSize(size_t *DoNotOptimize, uint32_t *DoNotOptimize32) {
  DoNotOptimize[0] = __spirv_BuiltInSubgroupMaxSize();
  DoNotOptimize32[0] = __spirv_BuiltInSubgroupMaxSize() + 9;
  // CHECK-LABEL: @{{.*}}kernel_SubgroupMaxSize
  // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
  // CHECK: store i64 [[ZEXT0]]
  // CHECK: add i32 1, 9
}
