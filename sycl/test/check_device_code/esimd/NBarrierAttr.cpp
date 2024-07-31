// RUN: %clangxx -O2 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// Checks NBarrier ESIMD intrinsic translation.

#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL ESIMD_NOINLINE void callee(int x) {
  __ESIMD_NS::named_barrier_init<13>();
}

// inherits SLMSize and NBarrierCount from callee
void caller_abc(int x) {
  kernel<class kernel_abc>([=]() SYCL_ESIMD_KERNEL { callee(x); });
  // CHECK: define dso_local spir_kernel void @_ZTSZ10caller_abciE10kernel_abc() local_unnamed_addr #[[ATTR1:[0-9]+]]
}

// inherits only NBarrierCount from callee
void caller_xyz(int x) {
  kernel<class kernel_xyz>([=]() SYCL_ESIMD_KERNEL {
    callee(x);
    auto y = __ESIMD_ENS::named_barrier_allocate<35>();
    __ESIMD_NS::named_barrier_wait(y);
  });
  // CHECK: define dso_local spir_kernel void @_ZTSZ10caller_xyziE10kernel_xyz() local_unnamed_addr #[[ATTR2:[0-9]+]]
  // CHECK: call void @llvm.genx.nbarrier(i8 0, i8 13, i8 0)
}

// CHECK: attributes #[[ATTR1]] = { {{.*}} "VCNamedBarrierCount"="13"
// CHECK: attributes #[[ATTR2]] = { {{.*}} "VCNamedBarrierCount"="48"
