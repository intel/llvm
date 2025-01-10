// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// REQUIRES: aspect-usm_shared_allocations

// Checks that restrict annotated_ptr works in device code.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;

  auto Ptr = sycl::malloc_shared<int>(1, Q);
  syclexp::annotated_ptr<int, decltype(syclexp::properties(syclexp::restrict))>
      AnnotPtr{Ptr};
  Q.submit([&](sycl::handler &CGH) {
     CGH.single_task([=]() { *AnnotPtr = 42; });
   }).wait();
  assert(*Ptr == 42);
  free(Ptr, Q);

  return 0;
}

// CHECK-IR: spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %_arg_AnnotPtr)
