//==------- Windows_build_test.cpp - DPC++ ESIMD build test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: windows
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// The tests validates an ability to build ESIMD code on windows platform

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

class Kernel;

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();
  std::cout << "Device: " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  int *buffer = (int *)sycl::aligned_alloc_device(128, 1024, q);

  q.parallel_for<Kernel>(1, [=](sycl::item<1> it) SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    using namespace sycl::ext::intel::experimental::esimd;

    simd<int, 32> blk;
    simd<sycl::ext::oneapi::bfloat16, 16> A;
    simd<sycl::ext::oneapi::bfloat16, 256> B;
    simd<float, 16> C;
    lzd<uint>(blk);
    lzd<uint>(35);
    sycl::ext::intel::esimd::xmx::dpas<8, 1, float>(C, B, A);
    lsc_block_store<int, 32>(buffer, blk);
  });

  q.wait();
  sycl::free(buffer, q);
  std::cout << "Done" << std::endl;
  return 0;
}
