// TODO: Enable compilation w/o -fno-sycl-early-optimizations option.
// See https://github.com/intel/llvm/issues/2264 for more details.

// RUN: %clangxx -fsycl -fno-sycl-early-optimizations %s -o %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==----------------- fpga_lsu.cpp - SYCL FPGA LSU test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

// TODO: run is disabled, since no support added in FPGA backend yet. Check
// implementation correctness from CXX and SYCL languages perspective.

int test_lsu(cl::sycl::queue Queue) {
  int output_data[2];
  for (size_t i = 0; i < 2; i++) {
    output_data[i] = -1;
  }

  int input_data[2];
  for (size_t i = 0; i < 2; i++) {
    input_data[i] = i + 1;
  }

  {
    cl::sycl::buffer<int, 1> output_buffer(output_data, 1);
    cl::sycl::buffer<int, 1> input_buffer(input_data, 1);

    Queue.submit([&](cl::sycl::handler &cgh) {
      auto output_accessor =
          output_buffer.get_access<cl::sycl::access::mode::write>(cgh);
      auto input_accessor =
          input_buffer.get_access<cl::sycl::access::mode::read>(cgh);

      cgh.single_task<class kernel>([=] {
        auto input_ptr = input_accessor.get_pointer();
        auto output_ptr = output_accessor.get_pointer();

        using PrefetchingLSU =
            cl::sycl::intel::lsu<cl::sycl::intel::prefetch<true>,
                                 cl::sycl::intel::statically_coalesce<false>>;

        using BurstCoalescedLSU =
            cl::sycl::intel::lsu<cl::sycl::intel::burst_coalesce<true>,
                                 cl::sycl::intel::statically_coalesce<false>>;

        using CachingLSU =
            cl::sycl::intel::lsu<cl::sycl::intel::burst_coalesce<true>,
                                 cl::sycl::intel::cache<1024>,
                                 cl::sycl::intel::statically_coalesce<false>>;

        using PipelinedLSU = cl::sycl::intel::lsu<>;

        int X = PrefetchingLSU::load(input_ptr); // int X = input_ptr[0]
        int Y = CachingLSU::load(input_ptr + 1); // int Y = input_ptr[1]

        BurstCoalescedLSU::store(output_ptr, X); // output_ptr[0] = X
        PipelinedLSU::store(output_ptr + 1, Y);  // output_ptr[1] = Y
      });
    });
  }

  for (int i = 0; i < 2; i++) {
    if (output_data[i] != input_data[i]) {
      std::cout << "Unexpected read from output_data: " << output_data[i]
                << ", v.s. expected " << input_data[i] << std::endl;

      return -1;
    }
  }
  return 0;
}

int main() {
  cl::sycl::queue Queue{cl::sycl::intel::fpga_emulator_selector{}};

  return test_lsu(Queue);
}
