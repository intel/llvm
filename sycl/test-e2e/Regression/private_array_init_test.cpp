// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- private_array_init_test.cpp - Regression test for private array init -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

namespace s = sycl;

class A {
public:
  A() : _arr{0, 0, 0} {}

  size_t size() { return sizeof(_arr) / sizeof(_arr[0]); }

private:
  size_t _arr[3];
};

int main() {
  size_t data = 1;

  s::queue q;
  {
    s::buffer<size_t, 1> buf(&data, s::range<1>(1));
    q.submit([&](s::handler &cgh) {
      auto acc = buf.get_access<s::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() {
        // Test that SYCL program is not crashed if it contains a class/struct
        // that has private array member which is initialized by zeroes.
        acc[0] += A().size();
      });
    });
  }

  assert(data == 4);

  return 0;
}
