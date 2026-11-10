// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: sg-16 || sg-32

// UNSUPPORTED: native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22772

// Build and run once per sub-group size the device reports as supported.
// RUN: %if sg-16 %{ %{build} -DSG_SIZE=16 -o %t16.out && %{run} %t16.out %}
// RUN: %if sg-32 %{ %{build} -DSG_SIZE=32 -o %t32.out && %{run} %t32.out %}

// This test checks that a compile-time kernel property attached to a free
// function kernel (here sub_group_size) is honored when the kernel is launched
// through the enqueue functions that take a kernel_function_s (nd_launch /
// single_task). Previously the property was dropped because these functions
// submit a wrapper kernel that forwards to the free function kernel, and the
// property was not propagated to the wrapper. See the issue reported for the
// 2026.1 free function launch method. It also checks that the property is
// honored for a kernel specialization that SYCL_EXT_ONEAPI_KERNEL_FUNCTION
// selects by deducing a template argument from the launch arguments.

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <iostream>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

using sg_size_desc = sycl::info::kernel_device_specific::compile_sub_group_size;

template <int SIMD>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>)) void probe(
    int *ptr) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  ptr[item.get_global_linear_id()] =
      static_cast<int>(item.get_sub_group().get_local_linear_range());
}

template <int SIMD>
int check(const char *How, const int *Ptr, size_t N, size_t Expected) {
  for (size_t i = 0; i < N; ++i) {
    if (Ptr[i] != static_cast<int>(Expected)) {
      std::cout << "SIMD=" << SIMD << " (" << How << "): item " << i
                << " observed sub-group size " << Ptr[i] << ", expected "
                << Expected << std::endl;
      return 1;
    }
  }
  return 0;
}

template <int SIMD, typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::sub_group_size<SIMD>)) void probe_deduced(T *ptr) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  ptr[item.get_global_linear_id()] =
      static_cast<T>(item.get_sub_group().get_local_linear_range());
}

template <int SIMD> int test(sycl::queue &q) {
  constexpr size_t N = 64;

  // The sub-group size the device will actually pick for this kernel. This is
  // the source of truth: the property makes the compiler request exactly this
  // value, so a run through any launch method must agree with it.
  const size_t Expected =
      syclexp::get_kernel_info<probe<SIMD>, sg_size_desc>(q);

  int *Ptr = sycl::malloc_shared<int>(N, q);
  syclexp::nd_launch(q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{N}},
                     syclexp::kernel_function<probe<SIMD>>, Ptr);
  q.wait();
  int Ret = check<SIMD>("kernel_function", Ptr, N, Expected);

  // T = int is deduced from Ptr; SIMD cannot be deduced and is given.
  q.fill(Ptr, 0, N).wait();
  const size_t ExpectedDeduced =
      syclexp::get_kernel_info<probe_deduced<SIMD, int>, sg_size_desc>(q);
  syclexp::nd_launch(q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{N}},
                     SYCL_EXT_ONEAPI_KERNEL_FUNCTION(probe_deduced<SIMD>, Ptr));
  q.wait();
  Ret |=
      check<SIMD>("SYCL_EXT_ONEAPI_KERNEL_FUNCTION", Ptr, N, ExpectedDeduced);

  sycl::free(Ptr, q);
  return Ret;
}

int main() {
  sycl::queue q;
  int Ret = 0;

  Ret |= test<SG_SIZE>(q);

  return Ret;
}
