// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Offsets are deprecated, but we should still test that they work
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <sycl/sycl.hpp>

int main() {
  const sycl::range<3> global_range(8, 8, 8);
  const sycl::range<3> local_range(2, 2, 2);
  const sycl::id<3> global_offset(4, 4, 4);
  const sycl::nd_range<3> nd_range(global_range, local_range, global_offset);

  sycl::queue sycl_queue;
  const size_t elements_per_work_item = 6;
  int *ptr =
      sycl::malloc_shared<int>(global_range[0] * global_range[1] *
                                   global_range[2] * elements_per_work_item,
                               sycl_queue);

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class indexers>(nd_range, [ptr](sycl::nd_item<3> index) {
      int *wi_ptr = ptr + index.get_global_linear_id() * elements_per_work_item;

      wi_ptr[0] = index.get_global_id(0);
      wi_ptr[1] = index.get_global_id(1);
      wi_ptr[2] = index.get_global_id(2);

      wi_ptr[3] = index.get_local_id(0);
      wi_ptr[4] = index.get_local_id(1);
      wi_ptr[5] = index.get_local_id(2);
    });
  });
  return 0;
}
