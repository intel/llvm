// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests raw_kernel_arg with 32-bit sized scalars.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

auto constexpr CLSource = R"===(
__kernel void Kernel1(int in, __global int *out) {
  out[0] = in;
}

__kernel void Kernel2(float in, __global int *out) {
  out[0] = in;
}
)===";

int main() {
  sycl::queue Q;

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Q.get_context(),
          sycl::ext::oneapi::experimental::source_language::opencl, CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);

  int32_t *OutVals = sycl::malloc_shared<int32_t>(2, Q);
  int32_t IntVal = 42;
  float FloatVal = 3.12f;

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_arg(0, sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &IntVal, sizeof(int32_t)));
     CGH.set_arg(1, OutVals);
     CGH.single_task(ExecKB.ext_oneapi_get_kernel("Kernel1"));
   }).wait();

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_arg(0, sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &FloatVal, sizeof(float)));
     CGH.set_arg(1, OutVals + 1);
     CGH.single_task(ExecKB.ext_oneapi_get_kernel("Kernel2"));
   }).wait();

  assert(OutVals[0] == IntVal);
  assert(OutVals[1] == static_cast<int32_t>(FloatVal));

  sycl::free(OutVals, Q);
  return 0;
}
