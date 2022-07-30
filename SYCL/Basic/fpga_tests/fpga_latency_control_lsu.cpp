//==- fpga_latency_control_lsu.cpp - SYCL FPGA latency control on LSU test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, accelerator
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

using PrefetchingLSU = ext::intel::experimental::lsu<
    ext::intel::experimental::prefetch<true>,
    ext::intel::experimental::statically_coalesce<false>>;

using BurstCoalescedLSU = ext::intel::experimental::lsu<
    ext::intel::experimental::burst_coalesce<true>,
    ext::intel::experimental::statically_coalesce<false>>;

int test_latency_control(queue Queue) {
  std::vector<float> input_data = {1.23f};
  std::vector<float> output_data = {.0f};

  {
    buffer input_buffer(input_data);
    buffer output_buffer(output_data);

    Queue.submit([&](handler &cgh) {
      auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);

      auto output_accessor = output_buffer.get_access<access::mode::write>(cgh);

      cgh.single_task<class kernel>([=] {
        auto in_ptr = input_accessor.get_pointer();
        auto out_ptr = output_accessor.get_pointer();

        float value = PrefetchingLSU::load(
            in_ptr, ext::oneapi::experimental::properties(
                        ext::intel::experimental::latency_anchor_id<0>));

        BurstCoalescedLSU::store(
            out_ptr, value,
            ext::oneapi::experimental::properties(
                ext::intel::experimental::latency_constraint<
                    0, ext::intel::experimental::latency_control_type::exact,
                    5>));
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
  queue Queue{ext::intel::fpga_emulator_selector{}};

  return test_latency_control(Queue);
}
