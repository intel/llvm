//==---------------- bitreverse.cpp - DPC++ ESIMD bitreverse test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// TODO: Enable when driver issue fixed
// UNSUPPORTED: gpu
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>
using namespace sycl;

int reverse_2bytes(int num) {
  num &= 0xffff;
  num = (num << 8) | (num >> 8);
  num &= 0xffff;
  return num;
}

int reverse_2bits(int num) {
  num &= 0x3;
  num = (num << 1) | (num >> 1);
  num &= 0x3;
  return num;
}

template <typename T> T reverse_bits(T v) {
  T s = sizeof(v) * 8;
  T mask = ~(T)0;
  while ((s >>= 1) > 0) {
    mask ^= (mask << s);
    v = ((v >> s) & mask) | ((v << s) & ~mask);
  }
  return v;
}

int main() {
  queue q;
  auto in_out = malloc_shared<uint16_t>(6, q);
  in_out[0] = 0x1234;
  in_out[1] = 0x2;
  in_out[2] = 0b01001011;
  const uint16_t expected_out[] = {0x3412, 0x1, 0b11010010};

  q.single_task([=]() SYCL_ESIMD_KERNEL {
     in_out[3] = reverse_2bytes(in_out[0]);
     in_out[4] = reverse_2bits(in_out[1]);
     in_out[5] = reverse_bits<uint8_t>(in_out[2]);
   }).wait();

  bool pass = true;
  for (int i = 0; i < 3; i++) {
    std::cout << "expected = " << std::hex << expected_out[i]
              << ", computed = " << in_out[i + 3] << std::endl;
    pass &= in_out[i + 3] == expected_out[i];
  }
  std::cout << "Test " << (pass ? "Passed" : "Failed") << std::endl;
  free(in_out, q);
  return !pass;
}
