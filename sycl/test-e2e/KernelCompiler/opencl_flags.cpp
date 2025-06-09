// REQUIRES: ocloc && (opencl || level_zero)
// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test ensures that the Kernel Compiler build option flags
// are passed all the way through to the final binary when using OpenCL C
// source.

#include <cmath>

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

const int N = 8;
const char *KernelCLSource = "__kernel void sqrt_test(__global float* A) {"
                             "    __private int x = get_global_id(0);"
                             "    __private int y = get_global_id(1);"
                             "    __private int w = get_global_size(1);"
                             "    __private int address = x * w + y;"
                             "    A[address] = sqrt(A[address]);"
                             "}";

int main(void) {
  // Only one device is supported at this time, so we limit the queue and
  // context to that.
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  bool ok =
      q.get_device().ext_oneapi_can_build(syclex::source_language::opencl);
  if (!ok) {
    std::cout << "Apparently this device does not support OpenCL C source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return 0;
  }

  auto kb_src = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, KernelCLSource);
  auto kb_exe =
      syclex::build(kb_src, syclex::properties{syclex::build_options(
                                "-cl-fp32-correctly-rounded-divide-sqrt")});
  sycl::kernel sqrt_test = kb_exe.ext_oneapi_get_kernel("sqrt_test");

  float *A = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; i++)
    A[i] = static_cast<float>(i) / N;

  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(A);
     sycl::nd_range ndr{{N}, {1}};
     cgh.parallel_for(ndr, sqrt_test);
   }).wait();

  for (int i = 0; i < N; i++) {
    float diff = A[i] - std::sqrt(static_cast<float>(i) / N);
    if (diff != 0.0) {
      printf("i:%d diff:%.2e\n", i, diff);
      return 1; // Error
    }
  }
  sycl::free(A, q);

  return 0;
}