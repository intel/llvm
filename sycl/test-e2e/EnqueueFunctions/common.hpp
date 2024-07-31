#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <iostream>

namespace oneapiext = sycl::ext::oneapi::experimental;

auto constexpr CLSource = R"===(
__kernel void KernelSingleTask(int num, int in, __global int *out) {
  for (size_t i = 0; i < num; ++i)
    out[i] = in;
}
__kernel void Kernel1D(int in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in;
}
__kernel void Kernel2D(int in, __global int *out) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t xs = get_global_size(0);
  size_t i = x + y * xs;
  out[i] = in;
}
__kernel void Kernel3D(int in, __global int *out) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);
  size_t xs = get_global_size(0);
  size_t ys = get_global_size(1);
  size_t i = x + y * xs + z * xs * ys;
  out[i] = in;
}
)===";

sycl::kernel_bundle<sycl::bundle_state::executable> CreateKB(sycl::queue Q) {
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> KBSource =
      oneapiext::create_kernel_bundle_from_source(
          Q.get_context(), oneapiext::source_language::opencl, CLSource);
  return oneapiext::build(KBSource, {Q.get_device()});
}

int Check(int *Data, int Expected, size_t Index, std::string TestName) {
  if (Data[Index] == Expected)
    return 0;
  std::cout << "Failed " << TestName << " at index " << Index << " : "
            << Data[Index] << " != " << Expected << std::endl;
  return 1;
}
