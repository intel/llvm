// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <cstdint>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<uint32_t> spec_const{1000};

int main() {
  queue myQueue;
  uint32_t out_val = 0;
  buffer<uint32_t, 1> out(&out_val, sycl::range<1>{1});
  myQueue.submit([&](handler &cgh) {
    accessor out_acc{out, cgh, write_only};

    cgh.set_specialization_constant<spec_const>(0);

    cgh.parallel_for<class SpecConstant>(
        out.get_range(), [=](item<1> item_id, kernel_handler h) {
          uint32_t spec_const_val = h.get_specialization_constant<spec_const>();
          out_acc[0] = spec_const_val;
        });
  });

  myQueue.wait();
  return 0;
}
