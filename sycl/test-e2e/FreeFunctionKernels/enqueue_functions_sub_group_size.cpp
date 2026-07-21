// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: Device incompatible error

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that a compile-time kernel property attached to a free
// function kernel (here sub_group_size) is honored when the kernel is launched
// through the enqueue functions that take a kernel_function_s (nd_launch /
// single_task). Previously the property was dropped because these functions
// submit a wrapper kernel that forwards to the free function kernel, and the
// property was not propagated to the wrapper. See the issue reported for the
// 2026.1 free function launch method.

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
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>))
void probe(int *ptr) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  ptr[item.get_global_linear_id()] =
      static_cast<int>(item.get_sub_group().get_local_linear_range());
}

template <int SIMD> int test(sycl::queue &q) {
  constexpr size_t N = 64;

  // The sub-group size the device will actually pick for this kernel. This is
  // the source of truth: the property makes the compiler request exactly this
  // value, so a run through any launch method must agree with it.
  const size_t Expected = syclexp::get_kernel_info<probe<SIMD>, sg_size_desc>(q);

  int *Ptr = sycl::malloc_shared<int>(N, q);
  syclexp::nd_launch(q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{N}},
                     syclexp::kernel_function<probe<SIMD>>, Ptr);
  q.wait();

  int Ret = 0;
  for (size_t i = 0; i < N; ++i) {
    if (Ptr[i] != static_cast<int>(Expected)) {
      std::cout << "SIMD=" << SIMD << ": item " << i << " observed sub-group "
                << "size " << Ptr[i] << ", expected " << Expected << std::endl;
      Ret = 1;
      break;
    }
  }
  sycl::free(Ptr, q);
  return Ret;
}

int main() {
  sycl::queue q;
  int Ret = 0;
  Ret |= test<16>(q);
  Ret |= test<32>(q);
  return Ret;
}
