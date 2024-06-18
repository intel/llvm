// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---------- unused_pointer.cpp - test pointers in struct --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==----------------------------------------------------------------------==//

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace std;

struct struct_with_pointer {
  int data;
  int *ptr1;             // Unused pointer
  float *ptr2;           // Unused pointer
  int *ptr_array1[2];    // Unused pointer array
  int *ptr_array2[2][3]; // Unused pointer array
};

int main(int argc, char **argv) {
  struct_with_pointer obj;
  obj.data = 10;
  int data = 0;
  sycl::queue queue;
  {
    sycl::buffer<int, 1> buf(&data, 1);
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() { acc[0] = obj.data; });
    });
  }
  if (data != 10) {
    printf("FAILED\ndata = %d\n", data);
    return 1;
  }
  return 0;
}
