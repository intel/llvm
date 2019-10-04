// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ vec_op.cpp - SYCL vec operations basic test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL_SIMPLE_SWIZZLES

#include <CL/sycl.hpp>

#include <cassert>

namespace s = cl::sycl;

template <typename> class test;

template <typename T> void testUnaryOp(const s::vec<T, 2> V) {
  s::buffer<int, 1> Buf(s::range<1>(4));
  s::queue Queue;
  Queue.submit([&](s::handler &Cgh) {
    auto Acc = Buf.get_access<s::access::mode::write>(Cgh);
    Cgh.single_task<test<T>>([=]() {
      Acc[0] = s::all(+V == s::vec<T, 2>{0.0, 1.0});
      Acc[1] = s::all(-V == s::vec<T, 2>{-0.0, -1});
      Acc[2] = s::all(+V.yx() == s::vec<T, 2>{1.0, 0.0});
      Acc[3] = s::all(-V.yx() == s::vec<T, 2>{-1.0, -0.0});
    });
  });
  auto Acc = Buf.get_access<s::access::mode::read>();
  assert(Acc[0] == true);
  assert(Acc[1] == true);
  assert(Acc[2] == true);
  assert(Acc[3] == true);
}

int main() {
  testUnaryOp(s::int2{0, 1});
  testUnaryOp(s::float2{0.f, 1.f});
  testUnaryOp(s::half2{0.0, 1.0});
  return 0;
}
