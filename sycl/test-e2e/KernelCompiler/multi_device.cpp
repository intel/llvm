// REQUIRES: (opencl || level_zero)
// RUN: %{build} -o %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 %{run} %t.out

#include <sycl/detail/core.hpp>

// Test to check that bundle is buildable from OpenCL source if there are
// multiple devices in the context.

auto constexpr CLSource = R"===(
__kernel void Kernel1(int in, __global int *out) {
  out[0] = in;
}

__kernel void Kernel2(short in, __global short *out) {
  out[0] = in;
}
)===";

int main() {
  sycl::platform Platform;
  auto Context = Platform.ext_oneapi_get_default_context();

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Context, sycl::ext::oneapi::experimental::source_language::opencl,
          CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);
  return 0;
}
