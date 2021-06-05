//==----------- same-kernel.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// >> ---- compile src1
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c %s -o %t-same-kernel-a.o
//
// >> ---- compile src2
// RUN: %clangxx -DB_CPP=1 -fsycl -fsycl-targets=%sycl_triple -c %s -o %t-same-kernel-b.o
//
// >> ---- link the full hetero app
// RUN: %clangxx %t-same-kernel-a.o %t-same-kernel-b.o -o %t-same-kernel.exe -fsycl -fsycl-targets=%sycl_triple

#include <CL/sycl.hpp>

using namespace cl::sycl;

class TestFnObj {
public:
  TestFnObj(buffer<int> &buf, handler &cgh) :
    data(buf.get_access<access::mode::write>(cgh)) {}
  accessor<int, 1, access::mode::write, access::target::global_buffer> data;
  void operator()(id<1> item) const { data[item] = item[0]; }
};

void kernel2();

#ifndef B_CPP
void kernel2() {
  static int data[256];
  {
    buffer<int> b(data, range<1>(256));
    queue q;
    q.submit([&](handler &cgh){
      TestFnObj kernel(b, cgh);
      cgh.parallel_for(range<1>(256), kernel);
    });
  }
  for (int i = 0; i < 256; i++) {
    assert(data[i] == i);
  }
}
#else // B_CPP
void kernel1() {
  static int data[10];
  {
    buffer<int> b(data, range<1>(10));
    queue q;
    q.submit([&](cl::sycl::handler &cgh){
      TestFnObj kernel(b, cgh);
      cgh.parallel_for(range<1>(10), kernel);
    });
  }
  for (int i = 0; i < 10; i++) {
    assert(data[i] == i);
  }
}

int main() {
  kernel1();
  kernel2();

  return 0;
}
#endif // B_CPP
