// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 %s

// An argument list held in a container converts to the std::span that the
// sequence overloads take, but a parameter pack is an exact match and wins
// overload resolution. Without the forwarding the pack overloads do, the
// container object itself would be bound as a single kernel argument, which
// compiles for any trivially copyable container and only fails once the kernel
// is launched, with a fault or an adapter error that says nothing about the
// argument list. Check that every spelling of an argument list is accepted by
// every launch function that takes the arguments of a sycl::kernel, and that a
// single raw_kernel_arg is still one argument.
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
namespace ext_detail = sycl::ext::oneapi::experimental::detail;
using oneapiext::raw_kernel_arg;

// Pins which single argument a parameter pack overload takes to be the argument
// list: a container of raw_kernel_arg in any value category, which is forwarded
// in C++20 and diagnosed before, but never one raw_kernel_arg.
template <typename... ArgsT>
constexpr bool IsArgList =
#if __cpp_lib_span
    ext_detail::is_arg_list_container_v<ArgsT...>;
#else
    ext_detail::is_arg_list_sequence_v<ArgsT...>;
#endif

static_assert(IsArgList<std::vector<raw_kernel_arg> &>);
static_assert(IsArgList<const std::vector<raw_kernel_arg> &>);
static_assert(IsArgList<std::vector<raw_kernel_arg>>);
static_assert(IsArgList<std::array<raw_kernel_arg, 2> &>);
static_assert(IsArgList<const std::array<raw_kernel_arg, 2> &>);
static_assert(IsArgList<std::array<raw_kernel_arg, 2>>);
static_assert(IsArgList<raw_kernel_arg (&)[2]>);
static_assert(IsArgList<const raw_kernel_arg (&)[2]>);
static_assert(IsArgList<sycl::span<raw_kernel_arg>>);
static_assert(IsArgList<sycl::span<const raw_kernel_arg>>);
#if __cpp_lib_span
static_assert(IsArgList<std::span<raw_kernel_arg>>);
static_assert(IsArgList<std::span<const raw_kernel_arg>>);
#endif
static_assert(!IsArgList<raw_kernel_arg>);
static_assert(!IsArgList<raw_kernel_arg &>);
static_assert(!IsArgList<const raw_kernel_arg &>);
static_assert(!IsArgList<std::vector<int> &>);
static_assert(!IsArgList<int *>);
// Two containers are two kernel arguments, not an argument list.
static_assert(
    !IsArgList<std::vector<raw_kernel_arg> &, std::vector<raw_kernel_arg> &>);

void argument_list_spellings(sycl::queue Q, sycl::handler &CGH,
                             sycl::range<1> Range, sycl::nd_range<1> NdRange,
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

  oneapiext::single_task(Q, Kernel, Vector);
  oneapiext::single_task(Q, Kernel, Array);
  oneapiext::single_task(Q, Kernel, Span);
  oneapiext::single_task(Q, Kernel, MutableSpan);
  oneapiext::single_task(Q, Kernel, SyclSpan);
  oneapiext::single_task(CGH, Kernel, Vector);
  oneapiext::single_task(CGH, Kernel, Array);
  oneapiext::single_task(CGH, Kernel, Span);
  oneapiext::single_task(CGH, Kernel, MutableSpan);
  oneapiext::single_task(CGH, Kernel, SyclSpan);

  oneapiext::parallel_for(Q, Range, Kernel, Vector);
  oneapiext::parallel_for(Q, Range, Kernel, Array);
  oneapiext::parallel_for(Q, Range, Kernel, Span);
  oneapiext::parallel_for(Q, Range, Kernel, MutableSpan);
  oneapiext::parallel_for(Q, Range, Kernel, SyclSpan);
  oneapiext::parallel_for(CGH, Range, Kernel, Vector);
  oneapiext::parallel_for(CGH, Range, Kernel, Array);
  oneapiext::parallel_for(CGH, Range, Kernel, Span);
  oneapiext::parallel_for(CGH, Range, Kernel, MutableSpan);
  oneapiext::parallel_for(CGH, Range, Kernel, SyclSpan);

  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel, Vector);
  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel, Array);
  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel, Span);
  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel,
                          MutableSpan);
  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel, SyclSpan);
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel, Vector);
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel, Array);
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel, Span);
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel,
                          MutableSpan);
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel,
                          SyclSpan);

  oneapiext::nd_launch(Q, NdRange, Kernel, Vector);
  oneapiext::nd_launch(Q, NdRange, Kernel, Array);
  oneapiext::nd_launch(Q, NdRange, Kernel, Span);
  oneapiext::nd_launch(Q, NdRange, Kernel, MutableSpan);
  oneapiext::nd_launch(Q, NdRange, Kernel, SyclSpan);
  oneapiext::nd_launch(CGH, NdRange, Kernel, Vector);
  oneapiext::nd_launch(CGH, NdRange, Kernel, Array);
  oneapiext::nd_launch(CGH, NdRange, Kernel, Span);
  oneapiext::nd_launch(CGH, NdRange, Kernel, MutableSpan);
  oneapiext::nd_launch(CGH, NdRange, Kernel, SyclSpan);

  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel, Vector);
  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel, Array);
  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel, Span);
  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel,
                       MutableSpan);
  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel, SyclSpan);
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel, Vector);
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel, Array);
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel, Span);
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel,
                       MutableSpan);
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel,
                       SyclSpan);
#endif

  // One raw_kernel_arg is one argument, and typed arguments are unaffected.
  int *Pointer = nullptr;
  oneapiext::single_task(Q, Kernel,
                         oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::single_task(CGH, Kernel, Pointer, Value);
  oneapiext::parallel_for(Q, Range, Kernel,
                          oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::parallel_for(CGH, Range, Kernel, Pointer, Value);
  oneapiext::parallel_for(Q, oneapiext::launch_config{Range}, Kernel,
                          oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::parallel_for(CGH, oneapiext::launch_config{Range}, Kernel, Pointer,
                          Value);
  oneapiext::nd_launch(Q, NdRange, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::nd_launch(CGH, NdRange, Kernel, Pointer, Value);
  oneapiext::nd_launch(Q, oneapiext::launch_config{NdRange}, Kernel,
                       oneapiext::raw_kernel_arg{&Value, sizeof(Value)});
  oneapiext::nd_launch(CGH, oneapiext::launch_config{NdRange}, Kernel, Pointer,
                       Value);
}
