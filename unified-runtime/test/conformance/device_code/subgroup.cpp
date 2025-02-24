// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

struct KernelFunctor {
  sycl::accessor<size_t, 1, sycl::access_mode::write> Acc;

  KernelFunctor(sycl::accessor<size_t, 1, sycl::access_mode::write> Acc)
      : Acc(Acc) {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<8>};
  }

  void operator()(sycl::nd_item<1> NdItem) const {
    auto SG = NdItem.get_sub_group();
    if (NdItem.get_global_linear_id() == 0) {
      Acc[0] = SG.get_local_linear_range();
    }
  }
};

int main() {
  sycl::queue myQueue;
  size_t output = 0;
  sycl::buffer output_buff(&output, sycl::range(1));

  myQueue.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{output_buff, cgh, sycl::write_only, sycl::no_init};
    cgh.parallel_for<class FixedSgSize>(sycl::nd_range<1>(8, 2),
                                        KernelFunctor{acc});
  });

  myQueue.wait();
  return 0;
}
