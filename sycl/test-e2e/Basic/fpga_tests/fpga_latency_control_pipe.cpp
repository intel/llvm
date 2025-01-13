//== fpga_latency_control_pipe.cpp - SYCL FPGA latency control on pipe test ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, accelerator
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

using Pipe1 = ext::intel::experimental::pipe<class PipeClass1, int, 8>;
using Pipe2 = ext::intel::experimental::pipe<class PipeClass2, int, 8>;

int test_latency_control(queue Queue) {
  std::vector<int> input_data = {1};
  std::vector<int> output_data = {0};

  {
    buffer input_buffer(input_data);
    buffer output_buffer(output_data);

    Queue.submit([&](handler &cgh) {
      auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);

      auto output_accessor = output_buffer.get_access<access::mode::write>(cgh);

      cgh.single_task<class kernel>([=] {
        Pipe1::write(input_accessor[0]);

        int value = Pipe1::read(ext::oneapi::experimental::properties(
            ext::intel::experimental::latency_anchor_id<0>));

        Pipe2::write(
            value,
            ext::oneapi::experimental::properties(
                ext::intel::experimental::latency_anchor_id<1>,
                ext::intel::experimental::latency_constraint<
                    0, ext::intel::experimental::latency_control_type::exact,
                    2>));

        output_accessor[0] = Pipe2::read();
      });
    });
  }

  if (output_data[0] != input_data[0]) {
    std::cout << "Unexpected read from output_data: " << output_data[0]
              << ", v.s. expected " << input_data[0] << std::endl;

    return -1;
  }
  return 0;
}

int main() {
  queue Queue{ext::intel::fpga_emulator_selector_v};

  return test_latency_control(Queue);
}
