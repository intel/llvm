// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the nd_launch overload that takes an already built sycl::kernel with
// every argument passed through raw_kernel_arg, the USM pointer included. That
// is what a caller which only knows the signature as sizes has to do.
//
// The pointer is passed through the pointer form of raw_kernel_arg. Passing the
// bytes of a pointer instead would bind it as a value argument, which only
// reaches the kernel on Level Zero: the OpenCL adapter passes a value argument
// to clSetKernelArg, which rejects a USM pointer with CL_INVALID_MEM_OBJECT,
// and the Native CPU adapter puts the address of its own copy of the bytes into
// the argument slot instead of the pointer itself. The RawKernelArg tests cover
// the byte form of a pointer, on Level Zero only.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

namespace syclext = sycl::ext::oneapi;

constexpr size_t N = 1024;
constexpr size_t WGSize = 8;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addScalars(int *Ptr, int A, int B) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[I] += A + B;
}

template <auto *Func> sycl::kernel getKernel(sycl::queue &Q) {
  auto Bundle =
      oneapiext::get_kernel_bundle<Func, sycl::bundle_state::executable>(
          Q.get_context());
  return Bundle.template ext_oneapi_get_kernel<Func>();
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};
  int *Memory = sycl::malloc_shared<int>(N, Q);
  sycl::nd_range<1> Ndr{sycl::range<1>{N}, sycl::range<1>{WGSize}};

  int Failed = 0;

  int A = 10, B = 20;
  sycl::kernel ScalarsKernel = getKernel<addScalars>(Q);
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(
      Q, Ndr, ScalarsKernel,
      oneapiext::raw_kernel_arg{&Memory, oneapiext::pointer_arg},
      oneapiext::raw_kernel_arg{&A, sizeof(A)},
      oneapiext::raw_kernel_arg{&B, sizeof(B)});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 30, I, "pointer and scalars through raw_kernel_arg");

  sycl::free(Memory, Q);
  return Failed;
}
