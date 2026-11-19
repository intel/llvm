// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

// Free-function launches must retain the legacy device-copyability exception
// that was applied to their arguments when they were captured by a lambda.
// expected-no-diagnostics

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <type_traits>
#include <utility>

namespace syclexp = sycl::ext::oneapi::experimental;

struct Offset {
  int Value;
};

struct Argument {
  int *Output;
  const Offset Off;
};

static_assert(!std::is_trivially_copyable_v<Argument>);
static_assert(!sycl::is_device_copyable_v<Argument>);
static_assert(std::is_trivially_copy_constructible_v<Argument>);
static_assert(std::is_trivially_destructible_v<Argument>);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void singleTaskKernel(Argument Arg) { *Arg.Output = Arg.Off.Value; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void ndRangeKernel(Argument Arg) { *Arg.Output = Arg.Off.Value; }

template <typename T>
void checkLaunches(sycl::queue Q, sycl::handler &CGH, T &&Arg) {
  const sycl::nd_range<1> Range{sycl::range<1>{1}, sycl::range<1>{1}};
  const syclexp::launch_config Config{
      Range, syclexp::properties{syclexp::work_group_scratch_size(0)}};

  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>,
                       std::forward<T>(Arg));
  syclexp::single_task(CGH, syclexp::kernel_function<singleTaskKernel>,
                       std::forward<T>(Arg));
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>,
                     std::forward<T>(Arg));
  syclexp::nd_launch(CGH, Range, syclexp::kernel_function<ndRangeKernel>,
                     std::forward<T>(Arg));
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeKernel>,
                     std::forward<T>(Arg));
  syclexp::nd_launch(CGH, Config, syclexp::kernel_function<ndRangeKernel>,
                     std::forward<T>(Arg));
}

void test(sycl::queue Q, sycl::handler &CGH, int *Output) {
  Argument Arg{Output, {1}};
  const Argument ConstArg{Output, {2}};
  checkLaunches(Q, CGH, Arg);
  checkLaunches(Q, CGH, ConstArg);
  checkLaunches(Q, CGH, Argument{Output, {3}});
}
