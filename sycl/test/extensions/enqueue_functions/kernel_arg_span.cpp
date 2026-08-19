// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 %s

// An argument list held in a container converts to the sycl::span that the
// nd_launch span overloads take, but a parameter pack is an exact match and
// wins overload resolution. Without the forwarding the pack overloads do, the
// container object itself would be bound as a single kernel argument, which
// compiles for any trivially copyable container and produces wrong results at
// run time. Check that every spelling of an argument list is accepted, and that
// a single raw_kernel_arg is still one argument.

#include <sycl/sycl.hpp>

#include <array>
#include <vector>
#if __cpp_lib_span
#include <span>
#endif

namespace oneapiext = sycl::ext::oneapi::experimental;

void argument_list_spellings(sycl::queue Q, sycl::handler &CGH,
                             sycl::nd_range<1> Range,
                             const sycl::kernel &Kernel) {
  int Value = 1;
  std::vector<oneapiext::raw_kernel_arg> Vector{{&Value, sizeof(Value)}};
  std::array<oneapiext::raw_kernel_arg, 1> Array{
      oneapiext::raw_kernel_arg{&Value, sizeof(Value)}};
  sycl::span<const oneapiext::raw_kernel_arg> Span{Vector.data(),
                                                   Vector.size()};
  sycl::span<oneapiext::raw_kernel_arg> MutableSpan{Vector.data(),
                                                    Vector.size()};

  oneapiext::nd_launch(Q, Range, Kernel, Vector);
  oneapiext::nd_launch(Q, Range, Kernel, Array);
  oneapiext::nd_launch(Q, Range, Kernel, Span);
  oneapiext::nd_launch(Q, Range, Kernel, MutableSpan);
  oneapiext::nd_launch(CGH, Range, Kernel, Vector);
  oneapiext::nd_launch(CGH, Range, Kernel, Array);
  oneapiext::nd_launch(CGH, Range, Kernel, Span);
  oneapiext::nd_launch(CGH, Range, Kernel, MutableSpan);
#if __cpp_lib_span
  std::span<const oneapiext::raw_kernel_arg> StdSpan{Vector.data(),
                                                     Vector.size()};
  oneapiext::nd_launch(Q, Range, Kernel, StdSpan);
  oneapiext::nd_launch(CGH, Range, Kernel, StdSpan);
#endif

  // One raw_kernel_arg is one argument, and typed arguments are unaffected.
  oneapiext::nd_launch(Q, Range, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::nd_launch(CGH, Range, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  int *Pointer = nullptr;
  oneapiext::nd_launch(Q, Range, Kernel, Pointer, Value);
  oneapiext::nd_launch(CGH, Range, Kernel, Pointer, Value);
}
