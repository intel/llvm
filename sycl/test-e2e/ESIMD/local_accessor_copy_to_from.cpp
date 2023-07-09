//==-- local_accessor_copy_to_from.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// TODO: Enable the test when GPU driver is ready/fixed.
// XFAIL: opencl || windows || gpu-intel-pvc
// TODO: add support for local_accessors to esimd_emulator.
// UNSUPPORTED: esimd_emulator
// The test checks functionality of the gather/scatter local
// accessor-based ESIMD intrinsics.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T, unsigned VL> bool test(queue q) {
  constexpr size_t size = VL;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL << "...\n";

  // The test is going to use size elements of T type.
  auto Dev = q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  if (DeviceSLMSize < size) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has!"
              << std::endl;
    return false;
  }

  T *A = new T[size];

  for (unsigned i = 0; i < size; ++i) {
    A[i] = static_cast<T>(i);
  }

  try {
    buffer<T, 1> buf(A, range<1>(size));
    nd_range<1> glob_range{range<1>{1}, range<1>{1}};

    q.submit([&](handler &cgh) {
       auto acc = buf.template get_access<access::mode::read_write>(cgh);
       auto LocalAcc = local_accessor<T, 1>(size, cgh);
       cgh.parallel_for(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;
         simd<T, VL> valsIn;
         simd<T, VL> valsOut;
         valsIn.copy_from(acc, 0);
         valsIn.copy_to(LocalAcc, 0);
         valsOut.copy_from(LocalAcc, 0);
         valsOut.copy_to(acc, 0);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < size; ++i) {
    T gold = static_cast<T>(i);

    if (A[i] != gold) {
      if (++err_cnt < 35) {
        std::cout << "failed at index " << i << ": " << A[i] << " != " << gold
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  }

  delete[] A;

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char, 1>(q);
  passed &= test<char, 2>(q);
  passed &= test<char, 4>(q);
  passed &= test<char, 8>(q);
  passed &= test<char, 16>(q);
  passed &= test<char, 32>(q);
  passed &= test<char, 64>(q);
  passed &= test<char, 128>(q);
  passed &= test<short, 1>(q);
  passed &= test<short, 2>(q);
  passed &= test<short, 4>(q);
  passed &= test<short, 8>(q);
  passed &= test<short, 16>(q);
  passed &= test<short, 32>(q);
  passed &= test<short, 64>(q);
  passed &= test<short, 128>(q);
  passed &= test<int, 1>(q);
  passed &= test<int, 2>(q);
  passed &= test<int, 4>(q);
  passed &= test<int, 8>(q);
  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<int, 64>(q);
  passed &= test<int, 128>(q);
  passed &= test<float, 1>(q);
  passed &= test<float, 2>(q);
  passed &= test<float, 4>(q);
  passed &= test<float, 8>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);
  passed &= test<float, 64>(q);
  passed &= test<float, 128>(q);
  return passed ? 0 : 1;
}
