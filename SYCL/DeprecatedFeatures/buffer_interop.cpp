// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -D__SYCL_INTERNAL_API -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---------------- buffer_interop.cpp - SYCL buffer basic test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <memory>

using namespace sycl;

int main() {
  bool Failed = false;
  {
    queue Queue;
    if (!Queue.is_host()) {
      std::vector<int> Data1(10, -1);
      std::vector<int> Data2(10, -2);
      {
        buffer<int, 1> BufferA(Data1.data(), range<1>(10));
        buffer<int, 1> BufferB(Data2);

        program Program(Queue.get_context());
        Program.build_with_source(
            "kernel void override_source(global int* Acc) "
            "{Acc[get_global_id(0)] = 0; }\n");
        sycl::kernel Kernel = Program.get_kernel("override_source");
        Queue.submit([&](handler &CGH) {
          auto AccA = BufferA.get_access<access::mode::read_write>(CGH);
          CGH.set_arg(0, AccA);
          auto AccB = BufferB.get_access<access::mode::read_write>(CGH);
          CGH.parallel_for(sycl::range<1>(10), Kernel);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++) {
        if (Data2[i] != -2) {
          std::cout << " Data2[" << i << "] is " << Data2[i] << " expected "
                    << -2 << std::endl;
          assert(false);
          Failed = true;
        }
      }
      for (int i = 0; i < 10; i++) {
        if (Data1[i] != 0) {
          std::cout << " Data1[" << i << "] is " << Data1[i] << " expected "
                    << 0 << std::endl;
          assert(false);
          Failed = true;
        }
      }
    }
  }

  {
    queue Queue;
    if (!Queue.is_host()) {
      std::vector<int> Data1(10, -1);
      std::vector<int> Data2(10, -2);
      {
        buffer<int, 1> BufferA(Data1.data(), range<1>(10));
        buffer<int, 1> BufferB(Data2);
        accessor<int, 1, access::mode::read_write, access::target::device,
                 access::placeholder::true_t>
            AccA(BufferA);
        accessor<int, 1, access::mode::read_write, access::target::device,
                 access::placeholder::true_t>
            AccB(BufferB);

        program Program(Queue.get_context());
        Program.build_with_source(
            "kernel void override_source_placeholder(global "
            "int* Acc) {Acc[get_global_id(0)] = 0; }\n");
        sycl::kernel Kernel = Program.get_kernel("override_source_placeholder");

        Queue.submit([&](handler &CGH) {
          CGH.require(AccA);
          CGH.set_arg(0, AccA);
          CGH.require(AccB);
          CGH.parallel_for(sycl::range<1>(10), Kernel);
        });
      } // Data is copied back
      for (int i = 0; i < 10; i++) {
        if (Data2[i] != -2) {
          std::cout << " Data2[" << i << "] is " << Data2[i] << " expected "
                    << -2 << std::endl;
          assert(false);
          Failed = true;
        }
      }
      for (int i = 0; i < 10; i++) {
        if (Data1[i] != 0) {
          std::cout << " Data1[" << i << "] is " << Data1[i] << " expected "
                    << 0 << std::endl;
          assert(false);
          Failed = true;
        }
      }
    }
  }

  return Failed;
}
