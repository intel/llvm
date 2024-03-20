//==--- double_conversion.cpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is basic test for testing conversion between double and unsigned int.

// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/detail/core.hpp>

#include <iostream>
#include <typeinfo>

namespace esimd = sycl::ext::intel::esimd;
template <typename T> int test(sycl::queue queue, double TestValue) {
  double input = TestValue;
  T output = 0;
  const T expected =
      static_cast<T>(input); // exact values can be stored in double type, no
                             // implementation-defined rounding here

  // Call simd conversion constructor
  // Expectation: data not changed
  {
    sycl::range<1> range(1);
    sycl::buffer<double, 1> buffer_in(&input, range);
    sycl::buffer<T, 1> buffer_out(&output, range);

    queue
        .submit([&](sycl::handler &cgh) {
          const auto in = buffer_in.get_access<sycl::access_mode::read,
                                               sycl::target::device>(cgh);
          const auto out =
              buffer_out.template get_access<sycl::access_mode::write,
                                             sycl::target::device>(cgh);

          cgh.single_task([=]() SYCL_ESIMD_KERNEL {
            esimd::simd<double, 1> source;
            source.copy_from(in, 0);

            esimd::simd<T, 1> destination(source);
            destination.copy_to(out, 0);
          });
        })
        .wait_and_throw();
  }

  std::cout << " Expected value: " << expected << "\n";
  std::cout << " Retrieved value: " << output << "\n";

  if (output == expected) {
    std::cout << "Test SUCCEED\n";
    return 0;
  } else {
    std::cout << "Test FAILED\n";
    return 1;
  }
}

int main(int, char **) {
  sycl::queue queue;

  auto dev = queue.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  int test_result = 0;

  test_result |= test<unsigned int>(queue, 2147483647);
  test_result |= test<unsigned int>(queue, 2147483647UL + 1);
  test_result |= test<unsigned int>(queue, 4294967295);

  test_result |= test<int>(queue, 2147483647);

  return test_result;
}
