// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 %s

// An argument list held in a container converts to the std::span that the
// nd_launch sequence overloads take, but a parameter pack is an exact match and
// wins overload resolution. Without the forwarding the pack overloads do, the
// container object itself would be bound as a single kernel argument, which
// compiles for any trivially copyable container and only fails once the kernel
// is launched, with a fault or an adapter error that says nothing about the
// argument list. Check that every spelling of an argument list is accepted, and
// that a single raw_kernel_arg is still one argument.
//
// The sequence overloads take a std::span, so they exist only in C++20 and
// later; the first RUN line checks that the parameter pack overloads are
// unaffected without it.

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
#if __cpp_lib_span
  std::vector<oneapiext::raw_kernel_arg> Vector{{&Value, sizeof(Value)}};
  std::array<oneapiext::raw_kernel_arg, 1> Array{
      oneapiext::raw_kernel_arg{&Value, sizeof(Value)}};
  std::span<const oneapiext::raw_kernel_arg> Span{Vector.data(), Vector.size()};
  std::span<oneapiext::raw_kernel_arg> MutableSpan{Vector.data(),
                                                   Vector.size()};
  // A sycl::span still converts, so a caller holding one does not have to
  // change how it stores its arguments.
  sycl::span<const oneapiext::raw_kernel_arg> SyclSpan{Vector.data(),
                                                       Vector.size()};

  oneapiext::nd_launch(Q, Range, Kernel, Vector);
  oneapiext::nd_launch(Q, Range, Kernel, Array);
  oneapiext::nd_launch(Q, Range, Kernel, Span);
  oneapiext::nd_launch(Q, Range, Kernel, MutableSpan);
  oneapiext::nd_launch(Q, Range, Kernel, SyclSpan);
  oneapiext::nd_launch(CGH, Range, Kernel, Vector);
  oneapiext::nd_launch(CGH, Range, Kernel, Array);
  oneapiext::nd_launch(CGH, Range, Kernel, Span);
  oneapiext::nd_launch(CGH, Range, Kernel, MutableSpan);
  oneapiext::nd_launch(CGH, Range, Kernel, SyclSpan);
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
