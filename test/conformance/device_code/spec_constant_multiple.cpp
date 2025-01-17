// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <cstdint>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<uint32_t> a_spec;
constexpr specialization_id<uint64_t> b_spec;
constexpr specialization_id<bool> c_spec;

int main() {
  queue myQueue;
  uint64_t out_val = 0;
  buffer<uint64_t, 1> out(&out_val, sycl::range<1>{1});

  myQueue.submit([&](handler &cgh) {
    accessor out_acc{out, cgh, write_only};

    cgh.set_specialization_constant<a_spec>(0);
    cgh.set_specialization_constant<b_spec>(0);
    cgh.set_specialization_constant<c_spec>(false);

    cgh.parallel_for<class SpecConstant>(
        out.get_range(), [=](item<1> item_id, kernel_handler h) {
          uint32_t a = h.get_specialization_constant<a_spec>();
          uint64_t b = h.get_specialization_constant<b_spec>();
          bool c = h.get_specialization_constant<c_spec>();
          if (c) {
            out_acc[0] = b - a;
          } else {
            out_acc[0] = b + a;
          }
        });
  });

  myQueue.wait();

  return 0;
}
