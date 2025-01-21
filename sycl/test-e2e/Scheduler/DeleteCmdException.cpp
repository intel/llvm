//==-------------------  DeleteCmdException.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: level_zero

// RUN: %{build} -o %t.out
// RUN:  %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>

void test_exception(sycl::queue &q, sycl::buffer<int, 1> &buf,
                    size_t workGroupSize) {

  try {
    // Illegal nd_range
    auto illegal_range = sycl::nd_range<1>{sycl::range<1>{workGroupSize * 2},
                                           sycl::range<1>{workGroupSize + 32}};

    // Will throw when submitted
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
       cgh.parallel_for(illegal_range, [=](sycl::nd_item<1> nd_item) {
         acc[nd_item.get_global_linear_id()] = 42; // will not be reached
       });
     }).wait_and_throw();
  } catch (const sycl::exception &e) {
    std::cout << "exception caught: " << e.code() << ":\t";
    std::cout << e.what() << std::endl;
  }
}

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  int maxWorkGroupSize =
      dev.get_info<sycl::info::device::max_work_group_size>();

  constexpr size_t NumWorkItems =
      2048; // this value is arbitrary since kernel is never run.
  std::vector<int> source(NumWorkItems, 0);
  {
    // Buffers with their own memory will have their memory release deferred,
    // while buffers backstopped by host memory will release when the buffer is
    // destroyed. This means there are two different paths we need to check to
    // ensure we are not leaking resources when encountering exceptions.

    // buffer with own memory
    sycl::buffer<int, 1> buf{sycl::range<1>{NumWorkItems}};

    // buffer backstopped by host memory
    sycl::buffer<int, 1> buf2{source.data(), sycl::range<1>{NumWorkItems}};

    test_exception(q, buf, maxWorkGroupSize);

    test_exception(q, buf2, maxWorkGroupSize);
  }

  return 0;
}