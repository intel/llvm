// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Vector %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Array %s 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsyntax-only -std=c++17 -DSEQUENCE=Span %s 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Vector %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Array %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DSEQUENCE=Span %s

// The overloads that take the arguments of a sycl::kernel as a sequence take a
// std::span, so they do not exist before C++20. A caller that passes its
// argument list as a container there would otherwise have the container bound
// as a single kernel argument, which compiles for any trivially copyable one
// and only fails once the kernel is launched, so it is diagnosed instead.

// CHECK: Passing the arguments of a sycl::kernel as a sequence requires C++20

#include <sycl/sycl.hpp>

#include <array>
#include <vector>

namespace oneapiext = sycl::ext::oneapi::experimental;

void argument_list_as_a_container(sycl::queue Q, sycl::handler &CGH,
                                  sycl::nd_range<1> Range,
                                  const sycl::kernel &Kernel) {
  int Value = 1;
  std::vector<oneapiext::raw_kernel_arg> Vector{{&Value, sizeof(Value)}};
  std::array<oneapiext::raw_kernel_arg, 1> Array{
      oneapiext::raw_kernel_arg{&Value, sizeof(Value)}};
  sycl::span<const oneapiext::raw_kernel_arg> Span{Vector.data(),
                                                   Vector.size()};

  oneapiext::nd_launch(Q, Range, Kernel, SEQUENCE);
  oneapiext::nd_launch(CGH, Range, Kernel, SEQUENCE);
}

void one_argument_at_a_time(sycl::queue Q, sycl::handler &CGH,
                            sycl::nd_range<1> Range,
                            const sycl::kernel &Kernel) {
  // A single raw_kernel_arg and typed arguments are one argument each, so they
  // are unaffected by the standard in use.
  int Value = 1;
  int *Pointer = nullptr;
  oneapiext::nd_launch(Q, Range, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::nd_launch(CGH, Range, Kernel, Pointer, Value);
}
