//==-- local_accessor_gather_scatter.cpp  - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the gather/scatter local
// accessor-based ESIMD intrinsics.

#include "esimd_test_utils.hpp"

using namespace sycl;

template <typename T, unsigned VL, unsigned STRIDE> bool test(queue q) {
  constexpr size_t size = VL;
  constexpr int MASKED_LANE = VL - 1;

  std::cout << "Testing T=" << typeid(T).name() << " VL=" << VL
            << " STRIDE=" << STRIDE << "...\n";

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
       auto LocalAcc = local_accessor<T, 1>(size * STRIDE, cgh);
       cgh.parallel_for(glob_range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
         using namespace sycl::ext::intel::esimd;
         simd<T, VL> valsIn;
         valsIn.copy_from(acc, 0);
         simd_mask<VL> pred = 1;
         pred[MASKED_LANE] = 0; // mask out the last lane
         LocalAcc[MASKED_LANE * STRIDE] = -1;
         simd<uint32_t, VL> offsets(0, STRIDE * sizeof(T));
         scatter<T, VL>(LocalAcc, offsets, valsIn, 0, pred);

         simd<T, VL> valsOut = gather<T, VL>(LocalAcc, offsets, 0);

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
    T gold = i != MASKED_LANE ? static_cast<T>(i) : static_cast<T>(-1);

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
  passed &= test<char, 8, 1>(q);
  passed &= test<char, 16, 3>(q);
  passed &= test<short, 8, 8>(q);
  passed &= test<short, 16, 1>(q);
  passed &= test<int, 8, 2>(q);
  passed &= test<int, 16, 1>(q);
  passed &= test<float, 8, 2>(q);
  passed &= test<float, 16, 1>(q);
  passed &= test<float, 32, 1>(q);
  passed &= test<float, 32, 4>(q);
  return passed ? 0 : 1;
}
