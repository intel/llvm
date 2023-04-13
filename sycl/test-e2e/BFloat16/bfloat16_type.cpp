// RUN: %if cuda %{%clangxx -fsycl -fsycl-targets=%sycl_triple -DUSE_CUDA_SM80=1 -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out %}
// RUN: %if cuda %{%GPU_RUN_PLACEHOLDER %t.out %}
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// TODO currently the feature isn't supported on FPGA.
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//
// Not currently supported on HIP.
// UNSUPPORTED: hip

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
