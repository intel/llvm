// REQUIRES: aspect-usm_shared_allocations
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

// expected-error@+1 {{free function can not be variadic template function}}
template<typename ...Ts>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void templated_variadic(Ts... args) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  auto sum = (0 + ... + args);
}

struct TestStruct {
  int a;
  float b;
};

namespace sycl {
template <> struct is_device_copyable<TestStruct>: std::false_type {
};
}

// expected-error@+1 {{}}
template<typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
}

template void templated(TestStruct, TestStruct);
