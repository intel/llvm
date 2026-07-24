// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -fno-sycl-unnamed-lambda -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// This test checks that free function kernels submitted through the queue
// overloads of single_task/nd_launch (i.e. the handler-bypassing direct
// submission path) pass their explicit arguments correctly. It exercises the
// argument kinds that can be constructed and passed without a handler on the
// direct path: USM pointers and plain std-layout scalar/struct data.
// Buffer/image accessors are not covered here because they must go through the
// handler overloads.

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr int SIZE = 16;

// Plain std-layout struct passed by value as a kernel argument.
struct Coeffs {
  int Scale;
  int Bias;
};

// USM pointer + plain scalar argument.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void addScalar(int *Ptr, int Value) { *Ptr += Value; }

// USM pointers + plain std-layout struct argument.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void affine(int *Src, int *Dst, Coeffs C) {
  size_t Gid = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Dst[Gid] = Src[Gid] * C.Scale + C.Bias;
}

int main() {
  sycl::queue Q;
  int *Src = sycl::malloc_shared<int>(SIZE, Q);
  int *Dst = sycl::malloc_shared<int>(SIZE, Q);

  for (int I = 0; I < SIZE; ++I)
    Src[I] = I;

  // single_task with a USM pointer and a plain scalar argument.
  Src[0] = 10;
  syclexp::single_task(Q, syclexp::kernel_function<addScalar>, Src, 5);
  Q.wait();
  assert(Src[0] == 15);

  for (int I = 0; I < SIZE; ++I)
    Src[I] = I;

  // nd_launch with USM pointers and a plain std-layout struct argument.
  Coeffs C{3, 7};
  syclexp::nd_launch(
      Q, sycl::nd_range<1>(sycl::range<1>(SIZE), sycl::range<1>(SIZE)),
      syclexp::kernel_function<affine>, Src, Dst, C);
  Q.wait();
  for (int I = 0; I < SIZE; ++I)
    assert(Dst[I] == Src[I] * C.Scale + C.Bias);

  sycl::free(Src, Q);
  sycl::free(Dst, Q);
  return 0;
}
