// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the nd_launch overloads that take the arguments of a sycl::kernel as a
// span of raw_kernel_arg, i.e. an argument list whose length is only known at
// run time. They have to bind the same arguments in the same order as the
// parameter pack overloads, on the queue and on the handler alike.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <vector>

namespace syclext = sycl::ext::oneapi;

static_assert(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS >= 2,
              "The span overloads require version 2 of the extension");

constexpr size_t N = 1024;
constexpr size_t WGSize = 8;

// A mixture of argument sizes, so that a wrong size or a wrong order shows up
// as a wrong result rather than as a silent pass.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addMixed(int *Ptr, int A, long B, float C, char D) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[I] += A + static_cast<int>(B) + static_cast<int>(C) + D;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void increment(int *Ptr) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[I] += 1;
}

template <auto *Func> sycl::kernel getKernel(sycl::queue &Q) {
  auto Bundle =
      oneapiext::get_kernel_bundle<Func, sycl::bundle_state::executable>(
          Q.get_context());
  return Bundle.template ext_oneapi_get_kernel<Func>();
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};
  sycl::kernel Kernel = getKernel<addMixed>(Q);

  int *Memory = sycl::malloc_shared<int>(N, Q);
  sycl::nd_range<1> Ndr{sycl::range<1>{N}, sycl::range<1>{WGSize}};

  int A = 1;
  long B = 20;
  float C = 300.0f;
  char D = 4;
  constexpr int Sum = 1 + 20 + 300 + 4;

  // The argument list is built at run time, which is the case these overloads
  // exist for. The pointer says that it is one, so that it is bound as a
  // pointer rather than as the bytes it is made of, which only the Level Zero
  // backend binds as a pointer.
  std::vector<oneapiext::raw_kernel_arg> Args;
  Args.emplace_back(&Memory, oneapiext::pointer_arg);
  Args.emplace_back(&A, sizeof(A));
  Args.emplace_back(&B, sizeof(B));
  Args.emplace_back(&C, sizeof(C));
  Args.emplace_back(&D, sizeof(D));
  sycl::span<const oneapiext::raw_kernel_arg> ArgSpan{Args.data(), Args.size()};

  int Failed = 0;

  // A run of launches through one span, so that an argument bound to storage
  // that does not outlive a single call would show up.
  constexpr int Launches = 8;
  Q.memset(Memory, 0, N * sizeof(int));
  for (int I = 0; I < Launches; ++I)
    oneapiext::nd_launch(Q, Ndr, Kernel, ArgSpan);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum * Launches, I, "span overload");

  // The parameter pack overload has to agree element for element.
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(
      Q, Ndr, Kernel,
      oneapiext::raw_kernel_arg{&Memory, oneapiext::pointer_arg},
      oneapiext::raw_kernel_arg{&A, sizeof(A)},
      oneapiext::raw_kernel_arg{&B, sizeof(B)},
      oneapiext::raw_kernel_arg{&C, sizeof(C)},
      oneapiext::raw_kernel_arg{&D, sizeof(D)});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "parameter pack overload");

  // And so does the handler form of the span overload.
  Q.memset(Memory, 0, N * sizeof(int));
  Q.submit([&](sycl::handler &CGH) {
     oneapiext::nd_launch(CGH, Ndr, Kernel, ArgSpan);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "handler form of the span overload");

  // A one element span is the boundary against the parameter pack overload,
  // which a single raw_kernel_arg selects instead.
  std::vector<oneapiext::raw_kernel_arg> OneArg{
      {&Memory, oneapiext::pointer_arg}};
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, getKernel<increment>(Q),
                       sycl::span<const oneapiext::raw_kernel_arg>{
                           OneArg.data(), OneArg.size()});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 1, I, "one element span");

  // A dependency the scheduler has to track forces the command group path,
  // which the span form has to take as well.
  {
    sycl::buffer<int, 1> Buf{sycl::range<1>{N}};
    Q.memset(Memory, 0, N * sizeof(int));
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::write_only, sycl::no_init};
      CGH.fill(Acc, 7);
    });
    oneapiext::nd_launch(Q, Ndr, Kernel, ArgSpan);
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory, Sum, I, "span form behind a buffer dependency");
  }

  sycl::free(Memory, Q);
  return Failed;
}
