// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---- bit_cast.cpp - SYCL bit_cast test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>

constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename To, typename From>
class BitCastKernel;

template <typename To, typename From>
To do_bit_cast(const From &from) {
  std::vector<To> vec(1);
  {
    sycl::buffer<To, 1> buf(vec.data(), 1);
    sycl::queue q;
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.template get_access<sycl_write>(cgh);
      cgh.single_task<class BitCastKernel<To, From>>([=]() {
        acc[0] = sycl::bit_cast<To>(from);
      });
    });
  }
  return vec[0];
}

template <typename To, typename From>
int test(const From &from, const To &expected) {
  auto actual = do_bit_cast<To>(from);
  if (actual != expected) {
    std::cerr << "FAIL: Actual which is " << actual << " != expected which is " << expected << "\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

int main() {
  int ReturnCode = 0;

  std::cout << "cl::sycl::half to unsigned short ...\n";
  ReturnCode += test(cl::sycl::half(1.0f), (unsigned short)(15360));

  std::cout << "unsigned short to cl::sycl::half ...\n";
  ReturnCode += test((unsigned short)(16384), cl::sycl::half(2.0f));

  std::cout << "cl::sycl::half to short ...\n";
  ReturnCode += test(cl::sycl::half(1.0f), short(15360));

  std::cout << "short to cl::sycl::half ...\n";
  ReturnCode += test(short(16384), cl::sycl::half(2.0f));

  std::cout << "int to float ...\n";
  ReturnCode += test(int(2), float(2.8026e-45));

  std::cout << "float to int ...\n";
  ReturnCode += test(float(-2.4), int(-1072064102));

  std::cout << "unsigned int to float ...\n";
  ReturnCode += test((unsigned int)(6), float(8.40779e-45));

  std::cout << "float to unsigned int ...\n";
  ReturnCode += test(float(-2.4), (unsigned int)(3222903194));

  return ReturnCode;
}