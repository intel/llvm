// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    size_t array_size = 16;
    cl::sycl::queue sycl_queue;
    uint32_t *src = cl::sycl::malloc_device<uint32_t>(array_size, sycl_queue);
    uint32_t *dst = cl::sycl::malloc_device<uint32_t>(array_size, sycl_queue);
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class cpy_and_mult_usm>(
            cl::sycl::range<1>{array_size},
            [src, dst](cl::sycl::item<1> itemId) {
                auto id = itemId.get_id(0);
                dst[id] = src[id] * 2;
            });
    });
    return 0;
}
