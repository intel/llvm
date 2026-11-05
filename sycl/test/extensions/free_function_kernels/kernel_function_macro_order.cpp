// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

// The host side sees the forward declarations that the integration header
// provides for every free function kernel ahead of the application's code. A
// launch that selects a kernel only through them is rejected, because the
// device side, which has no integration header, would select another kernel.
// The expected diagnostics are host-only; the device side accepts both
// launches.

#ifdef __SYCL_DEVICE_ONLY__
// expected-no-diagnostics
#endif

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// A better-matching overload declared after the launch.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void late(int *p, double v) {}

void launch_late(sycl::queue q, int *p) {
#ifndef __SYCL_DEVICE_ONLY__
  // expected-error@+2 {{host compilation selects free function kernel 'late' of type 'void (int *, float)', which is not visible during device compilation; declare the kernel before the launch}}
#endif
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(late, p, 1.0f));
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void late(int *p, float v) {}

// A better-matching overload declared before the launch but after the
// using-declaration, which names only the overloads declared before it.
namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void via_using(int *p, double v) {}
} // namespace ns

#ifndef __SYCL_DEVICE_ONLY__
// expected-note@+2 {{using declaration}}
#endif
using ns::via_using;

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void via_using(int *p, float v) {}
} // namespace ns

void launch_via_using(sycl::queue q, int *p) {
#ifndef __SYCL_DEVICE_ONLY__
  // expected-error@+2 {{host compilation selects free function kernel 'via_using' of type 'void (int *, float)', which is not visible during device compilation; declare the kernel before the launch}}
#endif
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(via_using, p, 1.0f));
}
