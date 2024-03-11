// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -x c++ %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that all LLVM-IR instructions that work with SPIR-V builtins
// are correctly translated into GenX counterparts (implemented in
// LowerESIMD.cpp)

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

size_t caller() {

  size_t DoNotOpt[1];
  uint32_t DoNotOpt32[1];
  size_t DoNotOptXYZ[3];

  sycl::queue().submit([&](sycl::handler &cgh) {
    auto DoNotOptimize = &DoNotOpt[0];
    auto DoNotOptimize32 = &DoNotOpt32[0];

    kernel<class kernel_SubgroupLocalInvocationId>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupLocalInvocationId();
      DoNotOptimize32[0] = __spirv_SubgroupLocalInvocationId() + 3;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupLocalInvocationId
    // CHECK: [[ZEXT0:%.*]] = zext i32 0 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 0, 3

    kernel<class kernel_SubgroupSize>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupSize();
      DoNotOptimize32[0] = __spirv_SubgroupSize() + 7;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupSize
    // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 1, 7

    kernel<class kernel_SubgroupMaxSize>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupMaxSize();
      DoNotOptimize32[0] = __spirv_SubgroupMaxSize() + 9;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupMaxSize
    // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 1, 9
  });
  return DoNotOpt[0];
}
