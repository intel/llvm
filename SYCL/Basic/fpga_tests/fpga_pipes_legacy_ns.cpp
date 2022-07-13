// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==-------- fpga_pipes_legacy_ns.cpp - SYCL FPGA pipes test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/sycl.hpp>

class some_nb_pipe;

// Test for simple non-blocking pipes in legacy namespace (cl::sycl::)
template <typename PipeName> int test_simple_nb_pipe(cl::sycl::queue Queue) {
  int data[] = {0};

  using Pipe = cl::sycl::pipe<PipeName, int>;

  cl::sycl::buffer<int, 1> readBuf(data, 1);
  Queue.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<class writer>([=]() {
      bool SuccessCode = false;
      do {
        Pipe::write(42, SuccessCode);
      } while (!SuccessCode);
    });
  });

  cl::sycl::buffer<int, 1> writeBuf(data, 1);
  Queue.submit([&](cl::sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.single_task<class reader>([=]() {
      bool SuccessCode = false;
      do {
        write_acc[0] = Pipe::read(SuccessCode);
      } while (!SuccessCode);
    });
  });

  auto readHostBuffer = writeBuf.get_access<cl::sycl::access::mode::read>();
  if (readHostBuffer[0] != 42) {
    std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
              << 42 << std::endl;

    return -1;
  }

  return 0;
}

int main() {
  cl::sycl::queue Queue;

  // Non-blocking pipes
  return test_simple_nb_pipe<some_nb_pipe>(Queue);
}
