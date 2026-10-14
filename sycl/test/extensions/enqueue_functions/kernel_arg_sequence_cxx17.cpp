// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Vector %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Array %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Span %s 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Vector %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Array %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Span %s

// Before C++20 a trivially copyable sequence of raw_kernel_arg would bind as
// one kernel argument, so nd_launch rejects a sequence.

// CHECK: Passing the arguments of a sycl::kernel as a sequence requires C++20

#include <sycl/sycl.hpp>

#include <array>
#include <vector>

namespace oneapiext = sycl::ext::oneapi::experimental;

void argument_list_as_a_container(sycl::queue Q, sycl::handler &CGH,
                                  sycl::nd_range<1> NdRange,
                                  const sycl::kernel &Kernel) {
  int Value = 1;
  std::vector<oneapiext::raw_kernel_arg> Vector{{&Value, sizeof(Value)}};
  std::array<oneapiext::raw_kernel_arg, 1> Array{
      oneapiext::raw_kernel_arg{&Value, sizeof(Value)}};
  sycl::span<const oneapiext::raw_kernel_arg> Span{Vector.data(),
                                                   Vector.size()};

  oneapiext::nd_launch(Q, NdRange, Kernel, SEQUENCE);
  oneapiext::nd_launch(CGH, NdRange, Kernel, SEQUENCE);
}

void one_argument_at_a_time(sycl::queue Q, sycl::handler &CGH,
                            sycl::nd_range<1> NdRange,
                            const sycl::kernel &Kernel) {
  // A single raw_kernel_arg and typed arguments are still accepted.
  int Value = 1;
  int *Pointer = nullptr;
  oneapiext::nd_launch(Q, NdRange, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::nd_launch(CGH, NdRange, Kernel, Pointer, Value);
}
