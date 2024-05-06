// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    size_t array_size = 16;
    cl::sycl::queue sycl_queue;
    std::vector<uint32_t> src(array_size, 1);
    std::vector<uint32_t> dst(array_size, 1);
    auto src_buff =
        cl::sycl::buffer<uint32_t>(src.data(), cl::sycl::range<1>(array_size));
    auto dst_buff =
        cl::sycl::buffer<uint32_t>(dst.data(), cl::sycl::range<1>(array_size));

    sycl_queue.submit([&](cl::sycl::handler &cgh) {
        auto src_acc = src_buff.get_access<cl::sycl::access::mode::read>(cgh);
        auto dst_acc = dst_buff.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<class cpy_and_mult>(
            cl::sycl::range<1>{array_size},
            [src_acc, dst_acc](cl::sycl::item<1> itemId) {
                auto id = itemId.get_id(0);
                dst_acc[id] = src_acc[id] * 2;
            });
    });
    return 0;
}
