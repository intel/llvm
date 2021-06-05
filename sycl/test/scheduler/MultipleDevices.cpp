// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=host %t.out

//===- MultipleDevices.cpp - Test checking multi-device execution --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

int multidevice_test(queue MyQueue1, queue MyQueue2) {
  const size_t N = 100;

  buffer<int, 1> BufA(range<1>{N});
  buffer<int, 1> BufB(range<1>{N});
  buffer<int, 1> BufC(range<1>{N});
  buffer<int, 1> BufD(range<1>{N});

  MyQueue1.submit([&](handler &cgh) {
    auto A = BufA.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class init_a>(
        range<1>{N}, [=](id<1> index) { A[index[0]] = index[0]; });
  });

  MyQueue2.submit([&](handler &cgh) {
    auto B = BufB.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class init_b>(
        range<1>{N}, [=](id<1> index) { B[index[0]] = N - index[0]; });
  });

  MyQueue2.submit([&](handler& cgh) {
    auto A = BufA.get_access<access::mode::read>(cgh);
    auto B = BufB.get_access<access::mode::read_write>(cgh);
    auto C = BufC.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class op1>(range<1>{N}, [=](id<1> index) {
      B[index[0]] = B[index[0]] + A[index[0]];
      C[index[0]] = B[index[0]] - index[0];
    });
  });

  MyQueue2.submit([&](handler &cgh) {
    auto D = BufD.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class init_d>(range<1>{N},
                                   [=](id<1> index) { D[index[0]] = 1; });
  });

  MyQueue1.submit([&](handler& cgh) {
    auto B = BufB.get_access<access::mode::read>(cgh);
    auto C = BufC.get_access<access::mode::read>(cgh);
    auto D = BufD.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class op2>(range<1>{N}, [=](id<1> index) {
      D[index[0]] = D[index[0]] + B[index[0]] - C[index[0]];
    });
  });

  auto FinalD = BufD.get_access<access::mode::read>();
  std::cout << "Result:" << std::endl;
  for (size_t i = 0; i < N; i++) {

    // A[index[0]] = index[0];
    int A = i;
    // B[index[0]] = N - index[0];
    int B = N - i;
    // B[index[0]] = B[index[0]] + A[index[0]];
    B = B + A;
    // C[index[0]] = B[index[0]] - index[0];
    int C = B - i;
    // D[index[0]] = 1;
    int D = 1;
    // D[index[0]] = D[index[0]] + B[index[0]] - C[index[0]];
    D = D + B - C;

    int Expected = D;

    if (FinalD[i] != D) {
      std::cout << "Wrong value for element " << i
                << " Expected: " << Expected << " Got: " << FinalD[i]
                << std::endl;
      return -1;
    }
  }

  std::cout << "Good computation!" << std::endl;
  return 0;
}

int main() {
  host_selector hostSelector;

  int Result = -1;
  try {
    queue MyQueue1(hostSelector);
    queue MyQueue2(hostSelector);
    Result &= multidevice_test(MyQueue1, MyQueue2);
  } catch(cl::sycl::runtime_error &) {
    std::cout << "Skipping host and host" << std::endl;
  }

  return Result;
}
