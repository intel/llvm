// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that arguments passed to the enqueue functions that take a
// kernel_function_s (nd_launch / single_task) are converted to the types of the
// corresponding free function kernel parameters, as required by the extension
// specification. Previously the arguments were passed to handler::set_args
// with the types the caller supplied, so passing e.g. a double to a float
// parameter set an argument of the wrong size and the launch failed with
// UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE.

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// A user-defined type which implicitly converts to the parameter type of the
// kernels below. The conversion has to be implicit: the enqueue functions are
// constrained on `is_invocable_v`, so a type whose conversion operator is
// `explicit` is rejected at the call site rather than converted, which
// sycl/test/extensions/free_function_kernels/enqueue_functions_arg_constraints.cpp
// checks.
struct ConvertibleToInt {
  int Value;
  operator int() const { return Value; }
};

enum Scale : short { Twice = 2 };

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float Start, float *Ptr) {
  size_t Gid = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[Gid] = Start + static_cast<float>(Gid);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void scale(int Factor, int *Ptr) { *Ptr = *Ptr * Factor; }

constexpr size_t SIZE = 8;

int main() {
  sycl::queue Q;
  float *FPtr = sycl::malloc_shared<float>(SIZE, Q);
  int *IPtr = sycl::malloc_shared<int>(1, Q);

  const sycl::nd_range<1> Range{sycl::range<1>(SIZE), sycl::range<1>(SIZE)};
  const syclexp::launch_config Config{Range};

  // A double literal passed to a float parameter.
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<iota>, 1.5, FPtr);
  Q.wait();
  for (size_t I = 0; I < SIZE; ++I)
    assert(FPtr[I] == 1.5f + static_cast<float>(I));

  // Same, through the launch_config and the handler overloads.
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<iota>, 2.5, FPtr);
  Q.wait();
  for (size_t I = 0; I < SIZE; ++I)
    assert(FPtr[I] == 2.5f + static_cast<float>(I));

  Q.submit([&](sycl::handler &CGH) {
     syclexp::nd_launch(CGH, Range, syclexp::kernel_function<iota>, 3.5, FPtr);
   }).wait();
  for (size_t I = 0; I < SIZE; ++I)
    assert(FPtr[I] == 3.5f + static_cast<float>(I));

  Q.submit([&](sycl::handler &CGH) {
     syclexp::nd_launch(CGH, Config, syclexp::kernel_function<iota>, 4.5, FPtr);
   }).wait();
  for (size_t I = 0; I < SIZE; ++I)
    assert(FPtr[I] == 4.5f + static_cast<float>(I));

  // A short lvalue passed to an int parameter.
  *IPtr = 3;
  short Factor = 5;
  syclexp::single_task(Q, syclexp::kernel_function<scale>, Factor, IPtr);
  Q.wait();
  assert(*IPtr == 15);

  // An enum with a short underlying type passed to an int parameter.
  Q.submit([&](sycl::handler &CGH) {
     syclexp::single_task(CGH, syclexp::kernel_function<scale>, Twice, IPtr);
   }).wait();
  assert(*IPtr == 30);

  // A user-defined type which converts to an int parameter.
  syclexp::single_task(Q, syclexp::kernel_function<scale>, ConvertibleToInt{10},
                       IPtr);
  Q.wait();
  assert(*IPtr == 300);

  sycl::free(FPtr, Q);
  sycl::free(IPtr, Q);
  return 0;
}
