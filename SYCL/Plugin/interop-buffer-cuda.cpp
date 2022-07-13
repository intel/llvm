// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--- interop-cuda.cpp - SYCL test for CUDA buffer interop API ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  int Data[1] = {0};
  sycl::buffer<int, 1> Buffer(&Data[0], sycl::range<1>(1));
  {
    sycl::queue Queue{sycl::gpu_selector()};
    Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buffer.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel>([=]() { (void)Acc; });
    });
  }

  auto NativeObj = sycl::get_native<sycl::backend::ext_oneapi_cuda>(Buffer);
  assert(NativeObj != 0);
}
