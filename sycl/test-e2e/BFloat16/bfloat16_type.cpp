// RUN: %if any-device-is-cuda %{ %{build} -DUSE_CUDA_SM80=1 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -o %t.out %}
// RUN: %if cuda %{ %{run} %t.out %}
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// TODO currently the feature isn't supported on FPGA.
// UNSUPPORTED: accelerator

//==----------- bfloat16_type.cpp - SYCL bfloat16 type test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bfloat16_type.hpp"

int main() {

#ifdef USE_CUDA_SM80
  // Special build for SM80 CUDA.
  sycl::device Dev{default_selector_v};
  if (Dev.get_platform().get_backend() != backend::ext_oneapi_cuda) {
    std::cout << "Test skipped; CUDA run was not run with CUDA device."
              << std::endl;
    return 0;
  }
  if (std::stof(Dev.get_info<sycl::info::device::backend_version>()) < 8.0f) {
    std::cout << "Test skipped; CUDA device does not support SM80 or newer."
              << std::endl;
    return 0;
  }
#endif

  return run_tests();
}
