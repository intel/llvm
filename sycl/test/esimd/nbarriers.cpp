// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void caller(int x) {
  kernel<class kernel_esimd>([=]() SYCL_ESIMD_KERNEL {
    named_barrier_init<7>();
    named_barrier_wait(2);
    // CHECK: call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33)
    // CHECK-NEXT: call spir_func void @_Z23__esimd_nbarrier_arrive{{.*}}
    named_barrier_signal(0, 0, 4, 4);
  });
}
