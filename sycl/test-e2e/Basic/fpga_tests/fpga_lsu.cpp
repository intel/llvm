//==----------------- fpga_lsu.cpp - SYCL FPGA LSU test --------------------==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: accelerator, opencl-aot
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

int test_lsu(sycl::queue Queue) {
  int output_data[2];
  for (size_t i = 0; i < 2; i++) {
    output_data[i] = -1;
  }

  int input_data[2];
  for (size_t i = 0; i < 2; i++) {
    input_data[i] = i + 1;
  }

  {
    sycl::buffer<int, 1> output_buffer(output_data, 1);
    sycl::buffer<int, 1> input_buffer(input_data, 1);

    Queue.submit([&](sycl::handler &cgh) {
      auto output_accessor =
          output_buffer.get_access<sycl::access::mode::write>(cgh);
      auto input_accessor =
          input_buffer.get_access<sycl::access::mode::read>(cgh);

      cgh.single_task<class kernel>([=] {
        auto input_ptr =
            input_accessor.get_multi_ptr<sycl::access::decorated::no>();
        auto output_ptr =
            output_accessor.get_multi_ptr<sycl::access::decorated::no>();

        using PrefetchingLSU =
            sycl::ext::intel::lsu<sycl::ext::intel::prefetch<true>,
                                  sycl::ext::intel::statically_coalesce<false>>;

        using BurstCoalescedLSU =
            sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                                  sycl::ext::intel::statically_coalesce<false>>;

        using CachingLSU =
            sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                                  sycl::ext::intel::cache<1024>,
                                  sycl::ext::intel::statically_coalesce<false>>;

        using PipelinedLSU = sycl::ext::intel::lsu<>;

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
  sycl::queue Queue{sycl::ext::intel::fpga_emulator_selector{}};

  return test_lsu(Queue);
}
