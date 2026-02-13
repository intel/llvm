//==--- aot.cpp - Simple vector addition (AOT compilation example) --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename T> class Vadd;

template <typename T, size_t N>
void vadd(const std::array<T, N> &A, const std::array<T, N> &B,
          std::array<T, N> &C) {
  sycl::queue Queue([](sycl::exception_list ExceptionList) {
    for (std::exception_ptr ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what();
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });

  sycl::range<1> numOfItems{N};
  sycl::buffer bufA(A.data(), numOfItems);
  sycl::buffer bufB(B.data(), numOfItems);
  sycl::buffer bufC(C.data(), numOfItems);

  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor accA{bufA, cgh, sycl::read_only};
    sycl::accessor accB{bufB, cgh, sycl::read_only};
    sycl::accessor accC{bufC, cgh, sycl::write_only};

    cgh.parallel_for<Vadd<T>>(numOfItems, [=](sycl::id<1> wiID) {
      accC[wiID] = accA[wiID] + accB[wiID];
    });
  });

  Queue.wait_and_throw();
}

int main() {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}}, B = {{1, 2, 3, 4}}, C;
  std::array<float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                E = {{1.f, 2.f, 3.f, 4.f}}, F;
  vadd(A, B, C);
  vadd(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "Incorrect result (element " << i << " is " << C[i] << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "Incorrect result (element " << i << " is " << F[i] << "!\n";
      return 1;
    }
  }
  std::cout << "Correct result!\n";
  return 0;
}
