// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------- fpga_pipes.cpp - SYCL FPGA pipes test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

// For pipes created with namespaces set
namespace some {
class pipe;
}

using namespace cl::sycl;

int main() {
  int data[] = {0};

  {
    // Test for non-blocking pipes
    queue Queue;
    using Pipe = pipe<class some_pipe, int, 1>;

    Queue.submit([&](handler &cgh) {
      cgh.single_task<class foo_nb>([=]() {
        bool SuccessCode = false;
        while (!SuccessCode)
          Pipe::write(42, SuccessCode);
      });
    });

    buffer<int, 1> writeBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      auto write_acc = writeBuf.get_access<access::mode::write>(cgh);

      cgh.single_task<class goo_nb>([=]() {
        bool SuccessCode = false;
        while (!SuccessCode)
          write_acc[0] = Pipe::read(SuccessCode);
      });
    });

    auto readHostBuffer = writeBuf.get_access<access::mode::read>();
    if (readHostBuffer[0] != 42) {
      std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
                << 42 << std::endl;

      return -1;
    }
  }

  {
    // Test for simple non-blocking pipes with explicit type
    queue Queue;

    buffer<int, 1> readBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      cgh.single_task<class boo_nb>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          pipe<class some_pipe, int, 1>::write(42, SuccessCode);
      });
    });

    buffer<int, 1> writeBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      auto write_acc = writeBuf.get_access<access::mode::write>(cgh);

      cgh.single_task<class zoo_nb>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          write_acc[0] = pipe<class some_pipe, int, 1>::read(SuccessCode);
      });
    });

    auto readHostBuffer = writeBuf.get_access<access::mode::read>();
    if (readHostBuffer[0] != 42) {
      std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
                << 42 << std::endl;

      return -1;
    }
  }

  {
    // Test for simple non-blocking pipes created with namespaces set
    queue Queue;

    buffer<int, 1> readBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      cgh.single_task<class foo_ns>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          pipe<class some::pipe, int, 1>::write(42, SuccessCode);
      });
    });

    buffer<int, 1> writeBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      auto write_acc = writeBuf.get_access<access::mode::write>(cgh);

      cgh.single_task<class boo_ns>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          write_acc[0] = pipe<class some::pipe, int, 1>::read(SuccessCode);
      });
    });

    auto readHostBuffer = writeBuf.get_access<access::mode::read>();
    if (readHostBuffer[0] != 42) {
      std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
                << 42 << std::endl;

      return -1;
    }
  }

  {
    // Test for forward declared pipes
    queue Queue;
    class pipe_type_for_lambdas;

    buffer<int, 1> readBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      cgh.single_task<class foo_la>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          pipe<class pipe_type_for_lambdas, int>::write(42, SuccessCode);
      });
    });

    buffer<int, 1> writeBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      cgh.single_task<class boo_la>([=]() {
        bool SuccessCode;
        while (!SuccessCode)
          pipe<class pipe_type_for_lambdas, int>::read(SuccessCode);
      });
    });

    auto readHostBuffer = writeBuf.get_access<access::mode::read>();
    if (readHostBuffer[0] != 42) {
      std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
                << 42 << std::endl;

      return -1;
    }
  }

  {
    // Test for blocking pipes
    queue Queue;
    using Pipe = pipe<class some_pipe, int, 1>;

    Queue.submit([&](handler &cgh) {
      cgh.single_task<class foo_b>([=]() { Pipe::write(42); });
    });

    buffer<int, 1> writeBuf(data, 1);
    Queue.submit([&](handler &cgh) {
      auto write_acc = writeBuf.get_access<access::mode::write>(cgh);

      cgh.single_task<class goo_b>([=]() { write_acc[0] = Pipe::read(); });
    });

    auto readHostBuffer = writeBuf.get_access<access::mode::read>();
    if (readHostBuffer[0] != 42) {
      std::cout << "Result mismatches " << readHostBuffer[0] << " Vs expected "
                << 42 << std::endl;

      return -1;
    }
  }

  return 0;
}
