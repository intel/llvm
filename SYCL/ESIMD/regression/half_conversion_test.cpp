// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//==- half_conversion_test.cpp - Test for half conversion under ESIMD_EMULATOR
// backend -==/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace ::sycl;
using namespace ::sycl::ext;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <int N>
using int_type_t = std::conditional_t<
    N == 1, int8_t,
    std::conditional_t<
        N == 2, int16_t,
        std::conditional_t<N == 4, int32_t,
                           std::conditional_t<N == 8, int64_t, void>>>>;

template <class Ty> bool test(queue q, int inc) {
  Ty *data = new Ty[1];

  data[0] = (Ty)0;
  Ty VAL = (Ty)inc;

  try {
    buffer<Ty, 1> buf(data, range<1>(1));
    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<::sycl::info::device::name>()
                << "\n";
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {
        simd<uint32_t, 1> offsets(0);
        simd<Ty, 1> vec = gather<Ty, 1>(acc, offsets);
        vec[0] += (Ty)inc;
        scalar_store<Ty>(acc, 0, vec[0]);
      });
    });
  } catch (::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] data;
    return false;
  }

  using Tint = int_type_t<sizeof(Ty)>;
  Tint ResBits = *(Tint *)&data[0];
  Tint GoldBits = *(Tint *)&VAL;

  std::cout << "Comparison of representation '" << inc << "' of Type "
            << typeid(Ty).name() << std::endl;
  std::cout << "Bits(data[0]) = 0x" << std::hex << ResBits << " / "
            << "Bits(GOLD) = 0x" << GoldBits << std::dec << std::endl;

  if (VAL == data[0]) {
    std::cout << "Pass";
  } else {
    std::cout << "Fail";
  }

  return ((Ty)inc == data[0]);
}

int main(int argc, char *argv[]) {
  bool passed = true;
  queue q;

  std::cout << "\n===================" << std::endl;
  passed &= test<short>(q, 1);
  std::cout << "\n===================" << std::endl;
  passed &= test<half>(q, 1);
  std::cout << "\n===================" << std::endl;
  passed &= test<float>(q, 1);
  std::cout << "\n===================" << std::endl;

  if (passed) {
    std::cout << "Pass!!" << std::endl;
  } else {
    std::cout << "Fail!!" << std::endl;
  }

  return passed ? 0 : -1;
}
