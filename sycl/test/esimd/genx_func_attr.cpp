// RUN: %clangxx -O2 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// Checks ESIMD intrinsic translation.

#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL ESIMD_NOINLINE void callee(int x) {
  slm_init<1234>();
  sycl::ext::intel::experimental::esimd::named_barrier_init<13>();
}

// inherits SLMSize and NBarrierCount from callee
void caller_abc(int x) {
  kernel<class kernel_abc>([=]() SYCL_ESIMD_KERNEL { callee(x); });
  // CHECK: define dso_local spir_kernel void @_ZTSZ10caller_abciE10kernel_abc(i32 noundef "VCArgumentIOKind"="0" %{{.*}}) local_unnamed_addr #2
}

// inherits only NBarrierCount from callee
void caller_xyz(int x) {
  kernel<class kernel_xyz>([=]() SYCL_ESIMD_KERNEL {
    slm_init(1235); // also works in non-O0
    callee(x);
  });
  // CHECK: define dso_local spir_kernel void @_ZTSZ10caller_xyziE10kernel_xyz(i32 noundef "VCArgumentIOKind"="0" %{{.*}}) local_unnamed_addr #3
}

// CHECK: attributes #2 = { {{.*}} "VCNamedBarrierCount"="13" "VCSLMSize"="1234"
// CHECK: attributes #3 = { {{.*}} "VCNamedBarrierCount"="13" "VCSLMSize"="1235"
