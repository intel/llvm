// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// REQUIRES: aspect-usm_shared_allocations

// Checks that restrict annotated_arg works in device code.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;

  int *Ptr = sycl::malloc_shared<int>(1, Q);
  syclexp::annotated_arg<int *,
                         decltype(syclexp::properties(syclexp::restrict))>
      AnnotArg{Ptr};
  Q.submit([&](sycl::handler &CGH) {
     CGH.single_task([=]() { *AnnotArg = 42; });
   }).wait();
  assert(*Ptr == 42);
  free(Ptr, Q);

  return 0;
}
