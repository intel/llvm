// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void kernel_esimd() {
  __ESIMD_NS::named_barrier_init<7>();
  __ESIMD_NS::named_barrier_wait(2);
  // CHECK: call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33)
  // CHECK-NEXT: call spir_func void @_Z23__esimd_nbarrier_arrive{{.*}}
  __ESIMD_NS::named_barrier_signal(0, 0, 4, 4);
}
