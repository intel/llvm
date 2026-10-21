// RUN: %clangxx -fsycl %s
// RUN: %clangxx -fsycl -fno-sycl-unnamed-lambda %s
// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-std=c++17" %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// This test serves as a sanity check that compilation succeeds
// for a simple free function kernel when using the host compiler
// flag with unnamed lambda extension enabled(default) and disabled.
// The free function kernel APIs other than SYCL_EXT_ONEAPI_KERNEL_FUNCTION need
// no host compiler support; that macro is used only where the host compiler
// supports it.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void kernel() {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void ping(int *x) {}

// The traits are specialized by the integration header, which only the host
// side includes.
#ifndef __SYCL_DEVICE_ONLY__
static_assert(syclexp::is_nd_range_kernel_v<kernel, 1>);
static_assert(syclexp::is_single_task_kernel_v<ping>);
static_assert(syclexp::is_kernel_v<ping>);
#endif

int main() {
  sycl::queue q;

  sycl::kernel_bundle bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  sycl::kernel_id kID =
      sycl::ext::oneapi::experimental::get_kernel_id<kernel>();
  sycl::kernel krn = bundle.get_kernel(kID);

  q.submit([&](sycl::handler &cgh) {
    sycl::nd_range<1> ndr(1, 1);
    cgh.parallel_for(ndr, krn);
  });

  int *ip = nullptr;
  syclexp::nd_launch(q, sycl::nd_range<1>(1, 1),
                     syclexp::kernel_function<kernel>);
  syclexp::single_task(q, syclexp::kernel_function<ping>, ip);
#ifdef SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ping, ip));
#endif
  return 0;
}
