// RUN: %{build} -fsycl-host-compiler=g++ -o %t.out
// REQUIRES: linux

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void kernel() {}

int main() {
  sycl::queue q;

  sycl::kernel_bundle bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  sycl::kernel_id kID =
      sycl::ext::oneapi::experimental::get_kernel_id<kernel>();
  sycl::kernel krn = bundle.get_kernel(kID);

  q.submit([&](sycl::handler &cgh) {
    sycl::nd_range<1> ndr;
    cgh.parallel_for(ndr, krn);
  });
  return 0;
}
