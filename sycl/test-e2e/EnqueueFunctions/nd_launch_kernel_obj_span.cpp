// REQUIRES: aspect-usm_shared_allocations

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} %{cpp20} -o %t.out
// RUN: %{run} %t.out

// Tests the nd_launch overloads that take the arguments of a sycl::kernel as a
// std::span of raw_kernel_arg.

#include <span>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <atomic>
#include <thread>
#include <vector>

namespace syclext = sycl::ext::oneapi;

static_assert(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS >= 2,
              "The span overloads require version 2 of the extension");

constexpr size_t N = 1024;
constexpr size_t WGSize = 8;

// A mixture of argument sizes, so that a wrong size or order shows up.
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

  std::vector<oneapiext::raw_kernel_arg> Args;
  Args.emplace_back(&Memory, oneapiext::pointer_arg);
  Args.emplace_back(&A, sizeof(A));
  Args.emplace_back(&B, sizeof(B));
  Args.emplace_back(&C, sizeof(C));
  Args.emplace_back(&D, sizeof(D));
  std::span<const oneapiext::raw_kernel_arg> ArgSpan{Args.data(), Args.size()};

  int Failed = 0;

  // Several launches through one span before a wait, each of which has to bind
  // the same arguments.
  constexpr int Launches = 8;
  Q.memset(Memory, 0, N * sizeof(int));
  for (int I = 0; I < Launches; ++I)
    oneapiext::nd_launch(Q, Ndr, Kernel, ArgSpan);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum * Launches, I, "span overload");

  // The same arguments through the parameter pack overload.
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

  Q.memset(Memory, 0, N * sizeof(int));
  Q.submit([&](sycl::handler &CGH) {
     oneapiext::nd_launch(CGH, Ndr, Kernel, ArgSpan);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "handler form of the span overload");

  // A container converts to std::span.
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, Kernel, Args);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "argument list passed as a container");

  Q.memset(Memory, 0, N * sizeof(int));
  Q.submit([&](sycl::handler &CGH) {
     oneapiext::nd_launch(CGH, Ndr, Kernel, Args);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "container through the handler form");

  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(
      Q, Ndr, Kernel,
      std::span<oneapiext::raw_kernel_arg>{Args.data(), Args.size()});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, Sum, I, "argument list as a mutable span");

  // A one element span is still an argument list, not a single argument.
  std::vector<oneapiext::raw_kernel_arg> OneArg{
      {&Memory, oneapiext::pointer_arg}};
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(
      Q, Ndr, getKernel<increment>(Q),
      std::span<const oneapiext::raw_kernel_arg>{OneArg.data(), OneArg.size()});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 1, I, "one element span");

  // An unfinished host task keeps the launch off the scheduler bypass, so the
  // arguments have to be copied; they change before the kernel runs.
  {
    std::atomic<bool> Release{false};
    Q.memset(Memory, 0, N * sizeof(int));
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&] {
        while (!Release.load())
          std::this_thread::yield();
      });
    });
    oneapiext::nd_launch(Q, Ndr, Kernel, ArgSpan);
    A = 0;
    B = 0;
    C = 0.0f;
    D = 0;
    Release = true;
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory, Sum, I, "span form behind a host task");
  }

  sycl::free(Memory, Q);
  return Failed != 0;
}
