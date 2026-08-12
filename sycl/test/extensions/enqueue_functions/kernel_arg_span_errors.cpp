// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// An argument list held in a container converts to the span that the nd_launch
// span overload takes, but a parameter pack is an exact match and wins overload
// resolution. Without a diagnostic the container object itself would be bound
// as a single kernel argument, which compiles for any trivially copyable
// wrapper and produces wrong results at run time.

#include <sycl/sycl.hpp>

#include <vector>

namespace oneapiext = sycl::ext::oneapi::experimental;

void argument_list_must_be_a_span(sycl::queue Q, sycl::nd_range<1> Range,
                                  const sycl::kernel &Kernel) {
  int Value = 1;
  std::vector<oneapiext::raw_kernel_arg> Args{{&Value, sizeof(Value)}};

  // expected-error@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{The kernel argument list must be passed as sycl::span<const raw_kernel_arg>}}
  oneapiext::nd_launch(Q, Range, Kernel, Args);

  sycl::span<oneapiext::raw_kernel_arg> Mutable{Args.data(), Args.size()};
  // expected-error@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{The kernel argument list must be passed as sycl::span<const raw_kernel_arg>}}
  oneapiext::nd_launch(Q, Range, Kernel, Mutable);

  // The spelling the diagnostic asks for, and a single argument that happens to
  // be a raw_kernel_arg, both have to keep working.
  sycl::span<const oneapiext::raw_kernel_arg> AsSpan{Args.data(), Args.size()};
  oneapiext::nd_launch(Q, Range, Kernel, AsSpan);
  oneapiext::nd_launch(Q, Range, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
}
