// REQUIRES: aspect-usm_shared_allocations

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} %{cpp20} -o %t.out
// RUN: %{run} %t.out

// Tests the single_task and parallel_for overloads, and the launch_config forms
// of parallel_for and nd_launch, that take the arguments of a sycl::kernel as a
// std::span of raw_kernel_arg, i.e. an argument list whose length is only known
// at run time. Those overloads take a std::span, hence C++20. They have to bind
// the same arguments in the same order as the parameter pack overloads, on the
// queue and on the handler alike.
// NOTE: This relies on the availability of an OpenCL C compiler.

#include <span>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <vector>

static_assert(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS >= 2,
              "These span overloads require version 2 of the extension");

constexpr size_t N = 1024;
constexpr size_t WGSize = 8;

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};

  if (!Q.get_device().ext_oneapi_can_build(
          oneapiext::source_language::opencl)) {
    std::cout
        << "Backend does not support OpenCL C source kernel bundle extension: "
        << Q.get_backend() << std::endl;
    return 0;
  }

  auto KB = CreateKB(Q);
  sycl::kernel KernelSingleTask = KB.ext_oneapi_get_kernel("KernelSingleTask");
  sycl::kernel Kernel1D = KB.ext_oneapi_get_kernel("Kernel1D");
  sycl::kernel Kernel2D = KB.ext_oneapi_get_kernel("Kernel2D");
  sycl::kernel Kernel3D = KB.ext_oneapi_get_kernel("Kernel3D");

  int *Memory = sycl::malloc_shared<int>(N, Q);

  int Count = N;
  int Value = 0;

  // A pointer argument has to say that it is one, since the byte form of a
  // pointer must not be passed to the byte overload of raw_kernel_arg.
  std::vector<oneapiext::raw_kernel_arg> SingleTaskArgs;
  SingleTaskArgs.emplace_back(&Count, sizeof(Count));
  SingleTaskArgs.emplace_back(&Value, sizeof(Value));
  SingleTaskArgs.emplace_back(&Memory, oneapiext::pointer_arg);
  std::span<const oneapiext::raw_kernel_arg> SingleTaskSpan{
      SingleTaskArgs.data(), SingleTaskArgs.size()};

  std::vector<oneapiext::raw_kernel_arg> Args;
  Args.emplace_back(&Value, sizeof(Value));
  Args.emplace_back(&Memory, oneapiext::pointer_arg);
  std::span<const oneapiext::raw_kernel_arg> ArgSpan{Args.data(), Args.size()};

  sycl::range<1> R1{N};
  sycl::range<2> R2{8, N / 8};
  sycl::range<3> R3{8, 8, N / 64};
  sycl::nd_range<1> Ndr1{R1, sycl::range<1>{WGSize}};
  sycl::nd_range<2> Ndr2{R2, sycl::range<2>{WGSize, WGSize}};

  int Failed = 0;

  // Each launch below passes a different value, so a launch that binds a value
  // left over from an earlier one, or binds the arguments in the wrong order,
  // shows up as a wrong result.
  auto Run = [&](int Written, const char *Name, auto &&Launch) {
    Q.memset(Memory, 0, N * sizeof(int));
    Value = Written;
    Launch();
    Q.wait();
    for (size_t I = 0; I < N; ++I)
      Failed += Check(Memory, Written, I, Name);
  };

  Run(41, "single_task span overload",
      [&] { oneapiext::single_task(Q, KernelSingleTask, SingleTaskSpan); });

  Run(42, "single_task span overload on the handler", [&] {
    Q.submit([&](sycl::handler &CGH) {
      oneapiext::single_task(CGH, KernelSingleTask, SingleTaskSpan);
    });
  });

  // The container holding the arguments converts to that span, so passing it
  // has to bind the arguments it holds rather than the container object.
  Run(43, "single_task argument list passed as a container",
      [&] { oneapiext::single_task(Q, KernelSingleTask, SingleTaskArgs); });

  // The parameter pack overload has to agree element for element.
  Run(44, "single_task parameter pack overload", [&] {
    oneapiext::single_task(
        Q, KernelSingleTask, oneapiext::raw_kernel_arg{&Count, sizeof(Count)},
        oneapiext::raw_kernel_arg{&Value, sizeof(Value)},
        oneapiext::raw_kernel_arg{&Memory, oneapiext::pointer_arg});
  });

  Run(45, "1D parallel_for span overload",
      [&] { oneapiext::parallel_for(Q, R1, Kernel1D, ArgSpan); });

  Run(46, "2D parallel_for span overload on the handler", [&] {
    Q.submit([&](sycl::handler &CGH) {
      oneapiext::parallel_for(CGH, R2, Kernel2D, ArgSpan);
    });
  });

  Run(47, "3D parallel_for argument list passed as a container",
      [&] { oneapiext::parallel_for(Q, R3, Kernel3D, Args); });

  Run(48, "1D parallel_for parameter pack overload", [&] {
    oneapiext::parallel_for(
        Q, R1, Kernel1D, oneapiext::raw_kernel_arg{&Value, sizeof(Value)},
        oneapiext::raw_kernel_arg{&Memory, oneapiext::pointer_arg});
  });

  Run(49, "1D parallel_for span overload with launch config", [&] {
    oneapiext::parallel_for(Q, oneapiext::launch_config{R1}, Kernel1D, ArgSpan);
  });

  Run(50, "2D parallel_for span overload with launch config on the handler",
      [&] {
        Q.submit([&](sycl::handler &CGH) {
          oneapiext::parallel_for(CGH, oneapiext::launch_config{R2}, Kernel2D,
                                  ArgSpan);
        });
      });

  Run(51, "3D parallel_for container with launch config", [&] {
    oneapiext::parallel_for(Q, oneapiext::launch_config{R3}, Kernel3D, Args);
  });

  Run(52, "1D nd_launch span overload with launch config", [&] {
    oneapiext::nd_launch(Q, oneapiext::launch_config{Ndr1}, Kernel1D, ArgSpan);
  });

  Run(53, "2D nd_launch span overload with launch config on the handler", [&] {
    Q.submit([&](sycl::handler &CGH) {
      oneapiext::nd_launch(CGH, oneapiext::launch_config{Ndr2}, Kernel2D,
                           ArgSpan);
    });
  });

  Run(54, "1D nd_launch container with launch config", [&] {
    oneapiext::nd_launch(Q, oneapiext::launch_config{Ndr1}, Kernel1D, Args);
  });

  Run(55, "1D nd_launch parameter pack overload with launch config", [&] {
    oneapiext::nd_launch(
        Q, oneapiext::launch_config{Ndr1}, Kernel1D,
        oneapiext::raw_kernel_arg{&Value, sizeof(Value)},
        oneapiext::raw_kernel_arg{&Memory, oneapiext::pointer_arg});
  });

  sycl::free(Memory, Q);
  return Failed;
}
