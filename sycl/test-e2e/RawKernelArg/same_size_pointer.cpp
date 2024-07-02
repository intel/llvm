// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests raw_kernel_arg with pointers and scalars to different 32-bit types.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

auto constexpr CLSource = R"===(
__kernel void Kernel1(int in, __global int *out) {
  out[0] = in;
}

__kernel void Kernel2(float in, __global float *out) {
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

  int32_t *IntOut = sycl::malloc_shared<int32_t>(1, Q);
  float *FloatOut = sycl::malloc_shared<float>(1, Q);
  int32_t IntVal = 42;
  float FloatVal = 3.12f;

  for (size_t I = 0; I < 2; ++I) {
    std::string KernelName = I == 0 ? "Kernel1" : "Kernel2";
    Q.submit([&](sycl::handler &CGH) {
       sycl::ext::oneapi::experimental::raw_kernel_arg KernelArg0 =
           I == 0 ? sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &IntVal, sizeof(int32_t))
                  : sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &FloatVal, sizeof(float));
       sycl::ext::oneapi::experimental::raw_kernel_arg KernelArg1 =
           I == 0 ? sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &IntOut, sizeof(int32_t *))
                  : sycl::ext::oneapi::experimental::raw_kernel_arg(
                        &FloatOut, sizeof(float *));

       CGH.set_arg(0, KernelArg0);
       CGH.set_arg(1, KernelArg1);
       CGH.single_task(ExecKB.ext_oneapi_get_kernel(KernelName));
     }).wait();
  }

  assert(IntOut[0] == IntVal);
  assert(FloatOut[0] == FloatVal);

  sycl::free(IntOut, Q);
  sycl::free(FloatOut, Q);
  return 0;
}
