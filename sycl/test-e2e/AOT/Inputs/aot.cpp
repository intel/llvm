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

template <typename T> class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N> &VA, const std::array<T, N> &VB,
                 std::array<T, N> &VC) {
  sycl::queue deviceQueue([](sycl::exception_list ExceptionList) {
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
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, [=](sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    });
  });

  deviceQueue.wait_and_throw();
}

int main() {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}}, B = {{1, 2, 3, 4}}, C;
  std::array<float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
