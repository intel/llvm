// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify=supported -Xclang -verify-ignore-unexpected=note,warning %s
// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -fsyntax-only -Xclang -verify=unsupported -Xclang -verify-ignore-unexpected=note,warning %s

// With -fsycl the header defines SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED and
// the macro launches kernels; without the builtin, including the header is
// valid but using the macro is an error at the call site.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void tick() {}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void fill(T *p, T v) {}

// The kernel property depends on a template parameter of the kernel.
template <int Dim, typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dim>))
void fill_dim(T *p, T v) {}

#ifdef SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED
// supported-no-diagnostics
static_assert(SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED == 1);
// Version 2 of the extension adds SYCL_EXT_ONEAPI_KERNEL_FUNCTION.
static_assert(SYCL_EXT_ONEAPI_FREE_FUNCTION_KERNELS >= 2);

void launch(sycl::queue q, sycl::nd_range<1> r, float *p) {
  // Zero launch arguments: the trailing comma must not be emitted.
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(tick));
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, p, 1.0f));
  syclexp::nd_launch(q, r,
                     SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill_dim<1>, p, 1.0f));
}
#else
void launch(float *p) {
  // clang-format off
  // unsupported-error@+1 {{undeclared identifier 'SYCL_EXT_ONEAPI_KERNEL_FUNCTION_requires_dpcpp_as_the_compiler'}}
  (void)SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, p, 1.0f);
  // clang-format on
}
#endif
