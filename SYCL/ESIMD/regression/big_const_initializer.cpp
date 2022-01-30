//==---------- big_const_initializer.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks that ESIMD program with big constant initializer list can
// compile and run correctly.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#define SIMD_WIDTH 16

#define N_SAMPLES (SIMD_WIDTH * 100)
#define N_PRINT 20

#define VAL1 0x9E3779B9
#define VAL2 0xBB67AE85

inline void
foo(sycl::ext::intel::experimental::esimd::simd<std::uint32_t, 16> &k) {
  sycl::ext::intel::experimental::esimd::simd<std::uint32_t, 16> k_add(
      {VAL1, VAL2, VAL1, VAL2, VAL1, VAL2, VAL1, VAL2, VAL1, VAL2, VAL1, VAL2,
       VAL1, VAL2, VAL1, VAL2});
  k += k_add;
}

int main(int argc, char **argv) {
  size_t nsamples = N_SAMPLES;
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  std::cout << "Running on "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "Driver version "
            << queue.get_device().get_info<sycl::info::device::driver_version>()
            << std::endl;

  sycl::buffer<std::uint32_t, 1> r(sycl::range<1>{nsamples});

  try {
    queue.submit([&](sycl::handler &cgh) {
      auto r_acc = r.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          sycl::range<1>{nsamples / SIMD_WIDTH},
          [=](sycl::item<1> item) SYCL_ESIMD_KERNEL {
            size_t id = item.get_id(0);
            sycl::ext::intel::experimental::esimd::simd<std::uint32_t, 16> key(
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            foo(key);
            key.copy_to(r_acc, id * SIMD_WIDTH * sizeof(std::uint32_t));
          });
    });
  } catch (const sycl::exception &e) {
    std::cout << "*** EXCEPTION caught: " << e.what() << "\n";
    return 1;
  }
  auto acc = r.template get_access<sycl::access::mode::read>();
  for (int i = 0; i < N_PRINT; i++) {
    std::cout << acc[i] << " ";
  }
  std::cout << std::endl;

  int err_cnt = 0;

  for (unsigned i = 0; i < nsamples; ++i) {
    uint32_t gold = i % 2 == 0 ? VAL1 + 1 : VAL2 + 1;
    uint32_t test = acc[i];
    if (test != gold) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << test << " != " << gold
                  << " (expected)"
                  << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(nsamples - err_cnt) / (float)nsamples) * 100.0f
              << "% (" << (nsamples - err_cnt) << "/" << nsamples << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
