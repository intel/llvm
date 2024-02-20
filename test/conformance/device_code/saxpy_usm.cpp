// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    size_t array_size = 16;

    cl::sycl::queue sycl_queue;
    uint32_t *X = cl::sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
    uint32_t *Y = cl::sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
    uint32_t *Z = cl::sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
    uint32_t A = 42;

    sycl_queue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class saxpy>(cl::sycl::range<1>{array_size},
                                      [=](cl::sycl::item<1> itemId) {
                                          auto i = itemId.get_id(0);
                                          Z[i] = A * X[i] + Y[i];
                                      });
    });
    return 0;
}
