// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests raw_kernel_arg in different combinations.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

constexpr size_t NumArgs = 4;

auto constexpr CLSource = R"===(
__kernel void Kernel(int in1, char in2, __global float *out, float in3) {
  out[0] = (float)in1 + (float)in2 + in3;
}
)===";

template <typename T>
void SetArg(sycl::handler &CGH, T &&Arg, size_t Index, size_t Iteration) {
  // Pick how we set the arg based on the bit at Index in Iteration.
  if (Iteration & (1 << Index))
    CGH.set_arg(Index, sycl::ext::oneapi::experimental::raw_kernel_arg(
                           &Arg, sizeof(T)));
  else
    CGH.set_arg(Index, Arg);
}

int main() {
  sycl::queue Q;

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Q.get_context(),
          sycl::ext::oneapi::experimental::source_language::opencl, CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);

  int Failed = 0;

  float *Out = sycl::malloc_shared<float>(1, Q);
  int32_t IntVal = 42;
  char CharVal = 100;
  float FloatVal = 1.23;

  float Expected =
      static_cast<float>(IntVal) + static_cast<float>(CharVal) + FloatVal;
  for (size_t I = 0; I < (2 >> (NumArgs - 1)); ++I) {
    Out[0] = 0.0f;
    Q.submit([&](sycl::handler &CGH) {
       SetArg(CGH, IntVal, 0, I);
       SetArg(CGH, CharVal, 1, I);
       SetArg(CGH, Out, 2, I);
       SetArg(CGH, FloatVal, 3, I);
       CGH.single_task(ExecKB.ext_oneapi_get_kernel("Kernel"));
     }).wait();

    if (Out[0] != Expected) {
      std::cout << "Failed for iteration " << I << ": " << Out[0]
                << " != " << Expected << std::endl;
      ++Failed;
    }
  }

  sycl::free(Out, Q);
  return Failed;
}
