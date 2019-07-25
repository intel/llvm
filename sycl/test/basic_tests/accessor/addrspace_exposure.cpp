// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
//==------- addrspace_exposure.cpp - SYCL accessor AS exposure test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <type_traits>

// This test checks that concrete address spaces are not exposed in device code
using namespace cl::sycl;

using cl::sycl::access::mode;
using cl::sycl::access::target;

int main() {
  range<1> Range(1);
  buffer<int, 1> GlobalBuf(Range);
  buffer<int, 1> ConstantBuf(Range);
  queue q;
  q.submit([&](handler &Cgh) {
    auto GlobalRWAcc =
        GlobalBuf.get_access<mode::read_write, target::global_buffer>(Cgh);
    auto GlobalRAcc =
        GlobalBuf.get_access<mode::read, target::global_buffer>(Cgh);
    auto ConstantAcc =
        ConstantBuf.get_access<mode::read, target::constant_buffer>(Cgh);
    accessor<int, 1, mode::read_write, target::local> LocalAcc(Range, Cgh);

    Cgh.single_task<class test>([=]() {
      static_assert(std::is_same<decltype(GlobalRWAcc[0]), int &>::value,
                    "Incorrect type from global read-write accessor");
      static_assert(std::is_same<decltype(GlobalRAcc[0]), int>::value,
                    "Incorrect type from global read accessor");
      static_assert(std::is_same<decltype(ConstantAcc[0]), int>::value,
                    "Incorrect type from constant accessor");
      static_assert(std::is_same<decltype(LocalAcc[0]), int &>::value,
                    "Incorrect type from local accessor");
    });
  });
}
