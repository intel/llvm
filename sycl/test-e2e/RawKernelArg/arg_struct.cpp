// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests raw_kernel_arg which is used to pass user-defined data types
// (structures) as kernel arguments.
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

constexpr size_t NumArgs = 2;

struct __attribute__((packed)) kernel_arg_t {
  int in1;
  char in2;
  float in3;
};

auto constexpr CLSource = R"===(
struct __attribute__((packed)) kernel_arg_t {
  int in1;
  char in2;
  float in3;
};

__kernel void Kernel(struct kernel_arg_t in, __global float4 *out) {
  out[0] = (float)in.in1 + (float)in.in2 + in.in3;
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
  kernel_arg_t InVal = {42, 100, 1.23};

  float Expected = static_cast<float>(InVal.in1) +
                   static_cast<float>(InVal.in2) + InVal.in3;
  for (size_t I = 0; I < (2 >> (NumArgs - 1)); ++I) {
    Out[0] = 0.0f;
    Q.submit([&](sycl::handler &CGH) {
       SetArg(CGH, InVal, 0, I);
       SetArg(CGH, Out, 1, I);
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


