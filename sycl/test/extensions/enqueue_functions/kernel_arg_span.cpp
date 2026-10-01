// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 %s

// From C++20 the parameter pack overloads forward a sequence of raw_kernel_arg
// to the std::span overloads.

#include <sycl/sycl.hpp>

#include <array>
#include <vector>
#if __cpp_lib_span
#include <span>
#endif

namespace oneapiext = sycl::ext::oneapi::experimental;
namespace ext_detail = sycl::ext::oneapi::experimental::detail;
using oneapiext::raw_kernel_arg;

// Both standards have to agree on what an argument list is: a sequence of
// raw_kernel_arg in any value category, never a single raw_kernel_arg.
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
// Two sequences do not form an argument list.
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
  // A sycl::span converts to std::span<const raw_kernel_arg> too.
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
