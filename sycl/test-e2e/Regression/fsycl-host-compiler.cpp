// RUN: %{build} -fsycl-host-compiler=g++ -DDEFINE_CHECK -fsycl-host-compiler-options="-DDEFINE_CHECK -std=c++17" -o %t.out
// RUN: %{run} %t.out
// REQUIRES: linux
//==------- fsycl-host-compiler.cpp - external host compiler test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Uses -fsycl-host-compiler=<compiler> on a simple test, requires 'g++'

#include <sycl/detail/core.hpp>

#ifndef DEFINE_CHECK
#error predefined macro not set
#endif // DEFINE_CHECK

using namespace sycl;

int main() {
  int data[] = {0, 0, 0};

  {
    buffer<int, 1> b(data, range<1>(3), {property::buffer::use_host_ptr()});
    queue q;
    q.submit([&](handler &cgh) {
      auto B = b.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class test>(range<1>(3), [=](id<1> idx) { B[idx] = 1; });
    });
  }

  bool isSuccess = true;

  for (int i = 0; i < 3; i++)
    if (data[i] != 1)
      isSuccess = false;

  {
    buffer<int, 1> b(1);
    queue q;
    q.submit([&](handler &cgh) {
       accessor a{b, cgh};
       cgh.single_task<class test2>([=] { a[0] = 42; });
     }).wait();
    host_accessor a{b};
    isSuccess &= (a[0] == 42);
  }

  if (!isSuccess)
    return -1;

  return 0;
}
