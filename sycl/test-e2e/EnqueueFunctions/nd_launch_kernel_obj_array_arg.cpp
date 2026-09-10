// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that an array argument reaches a `sycl::kernel` the same way through
// the handler-less nd_launch overload as it does through the handler.
//
// An array is bound as the bytes it is, which is what `handler::set_args` does
// with it, so the kernel below reads those 16 bytes as its parameter. The
// handler-less overload must not classify the array as a pointer argument: the
// runtime would then read the first bytes of the array as an address, which
// binds neither the bytes nor the array itself.
//
// Kernel arguments are sticky on the backend side and a bundle hands out the
// same underlying kernel for the same kernel id, so a correct launch would mask
// a wrong one. The handler-less path therefore goes first.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

namespace syclext = sycl::ext::oneapi;

constexpr size_t N = 8;

struct FourInts {
  int A, B, C, D;
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addFourInts(FourInts Values, int *Out) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Out[I] = Values.A + Values.B + Values.C + Values.D;
}

template <auto *Func> sycl::kernel getKernel(sycl::queue &Q) {
  auto Bundle =
      oneapiext::get_kernel_bundle<Func, sycl::bundle_state::executable>(
          Q.get_context());
  return Bundle.template ext_oneapi_get_kernel<Func>();
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};

  int *Out = sycl::malloc_shared<int>(N, Q);
  sycl::nd_range<1> Ndr{sycl::range<1>{N}, sycl::range<1>{N}};
  int Values[4] = {1, 2, 3, 4};
  constexpr int Expected = 1 + 2 + 3 + 4;

  int Failed = 0;

  sycl::kernel Kernel = getKernel<addFourInts>(Q);

  // The handler-less path, which must bind the array as plain bytes.
  Q.memset(Out, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, Kernel, Values, Out);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Out, Expected, I, "array without a handler");

  // The command group path has to agree with it.
  Q.memset(Out, 0, N * sizeof(int));
  Q.submit([&](sycl::handler &CGH) {
    oneapiext::nd_launch(CGH, Ndr, Kernel, Values, Out);
  });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Out, Expected, I, "array through the command group path");

  sycl::free(Out, Q);
  return Failed;
}
