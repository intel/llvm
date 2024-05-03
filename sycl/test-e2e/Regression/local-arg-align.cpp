// RUN: %{build} -o %t.out
//
// RUN: %{run} %t.out

// FIXME: enable opaque pointers support
// REQUIRES: TEMPORARY_DISABLED

//==-- local-arg-align.cpp - Test for local argument alignmnent ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <sycl/detail/core.hpp>

using namespace sycl;

// This test is a simple unit test to ensure that local kernel arguments are
// properly aligned.
int main(int argc, char *argv[]) {
  queue q;
  buffer<size_t, 1> res(2);

  q.submit([&](sycl::handler &h) {
     // Use two local buffers, one with an int and one with a float4
     local_accessor<cl_int, 1> a(1, h);
     local_accessor<float4, 1> b(1, h);

     auto ares = res.get_access<access::mode::read_write>(h);

     // Manually capture kernel arguments to ensure an order with the int
     // argument first and the float4 argument second. If the two arguments are
     // simply laid out consecutively, the float4 argument will not be
     // correctly aligned.
     h.parallel_for(sycl::nd_range<1>{1, 1}, [a, b, ares](sycl::nd_item<1>) {
       // Get the addresses of the two local buffers
       ares[0] = (size_t)&a[0];
       ares[1] = (size_t)&b[0];
     });
   }).wait_and_throw();

  auto hres = res.get_host_access();

  int ret = 0;
  // Check that the addresses are aligned as expected
  if (hres[0] % sizeof(cl_int) != 0) {
    std::cout
        << "Error: incorrect alignment for argument a, required alignment: "
        << sizeof(cl_int) << ", address: " << (void *)hres[0] << std::endl;
    ret = -1;
  }

  if (hres[1] % sizeof(float4) != 0) {
    std::cout
        << "Error: incorrect alignment for argument b, required alignment: "
        << sizeof(float4) << ", address: " << (void *)hres[1] << std::endl;
    ret = -1;
  }

  return ret;
}
