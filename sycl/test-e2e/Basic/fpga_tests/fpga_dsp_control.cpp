//==---------- fpga_dsp_control.cpp - SYCL FPGA DSP control test -----------==//
//
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

int test_dsp_control(sycl::queue Queue) {
  std::vector<float> input_data = {1.23f, 2.34f};
  std::vector<float> output_data = {.0f, .0f};

  {
    sycl::buffer input_buffer(input_data);
    sycl::buffer output_buffer(output_data);

    Queue.submit([&](sycl::handler &cgh) {
      auto input_accessor =
          input_buffer.get_access<sycl::access::mode::read>(cgh);

      auto output_accessor =
          output_buffer.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task<class kernel>([=] {
        sycl::ext::intel::math_dsp_control<sycl::ext::intel::Preference::DSP>(
            [&] { output_accessor[0] = input_accessor[0] + 1.0f; });

        sycl::ext::intel::math_dsp_control<sycl::ext::intel::Preference::DSP,
                                           sycl::ext::intel::Propagate::Off>(
            [&] { output_accessor[0] -= 1.0f; });

        sycl::ext::intel::math_dsp_control<
            sycl::ext::intel::Preference::Softlogic>(
            [&] { output_accessor[1] = input_accessor[1] + 1.0f; });

        sycl::ext::intel::math_dsp_control<
            sycl::ext::intel::Preference::Softlogic,
            sycl::ext::intel::Propagate::Off>(
            [&] { output_accessor[1] -= 1.0f; });
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
  sycl::queue Queue{sycl::ext::intel::fpga_emulator_selector_v};

  return test_dsp_control(Queue);
}
