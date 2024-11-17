// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests raw_kernel_arg which is used to pass OpenCL vector types as a special
// case of struct data types.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <CL/cl.h>

constexpr size_t NumArgs = 4;

auto constexpr CLSource = R"===(
__kernel void Kernel(int4 in1, char4 in2, __global float4 *out, float4 in3) {
  out[0] = convert_float4(in1) + convert_float4(in2) + in3;
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

  cl_float4 *Out = sycl::malloc_shared<cl_float4>(1, Q);
  cl_int4 IntVal = {42, 42, 42, 42};
  cl_char4 CharVal = {100, 100, 100, 100};
  cl_float4 FloatVal = {1.23, 1.23, 1.23, 1.23};

  float Expected = static_cast<float>(IntVal.s[0]) +
                   static_cast<float>(CharVal.s[0]) + FloatVal.s[0];
  for (size_t I = 0; I < (2 >> (NumArgs - 1)); ++I) {
    Out[0].s[I] = 0.0f;
    Q.submit([&](sycl::handler &CGH) {
       SetArg(CGH, IntVal, 0, I);
       SetArg(CGH, CharVal, 1, I);
       SetArg(CGH, Out, 2, I);
       SetArg(CGH, FloatVal, 3, I);
       CGH.single_task(ExecKB.ext_oneapi_get_kernel("Kernel"));
     }).wait();

    for (size_t Ind = 0; Ind < 4; ++Ind) {
      if (Out[0].s[Ind] != Expected) {
        std::cout << "Failed for iteration " << I << " at index " << Ind << ": "
                  << Out[0].s[Ind] << " != " << Expected << std::endl;
        ++Failed;
      }
    }
  }

  sycl::free(Out, Q);
  return Failed;
}

