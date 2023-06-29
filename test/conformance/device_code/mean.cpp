// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    const int array_size = 16;
    const int wg_size = 4;
    std::vector<uint32_t> in(array_size * wg_size, 1);
    std::vector<uint32_t> out(array_size, 0);
    cl::sycl::queue sycl_queue;
    auto in_buff = cl::sycl::buffer<uint32_t>(
        in.data(), cl::sycl::range<1>(array_size * wg_size));
    auto out_buff =
        cl::sycl::buffer<uint32_t>(out.data(), cl::sycl::range<1>(array_size));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
        sycl::local_accessor<uint32_t> local_mem(wg_size, cgh);
        auto in_acc = in_buff.get_access<cl::sycl::access::mode::read>(cgh);
        auto out_acc = out_buff.get_access<cl::sycl::access::mode::write>(cgh);

        cl::sycl::range<1> num_groups{array_size};
        cl::sycl::range<1> group_size{wg_size};
        cgh.parallel_for_work_group<class mean>(
            num_groups, group_size, [=](cl::sycl::group<1> group) {
                auto group_id = group.get_group_id();
                group.parallel_for_work_item([&](sycl::h_item<1> item) {
                    auto local_id = item.get_local_id(0);
                    auto in_index = (group_id * wg_size) + local_id;
                    local_mem[local_id] = in_acc[in_index];
                });
                cl::sycl::group_barrier(group);
                uint32_t total = 0;
                for (int i = 0; i < wg_size; i++) {
                    total += local_mem[i];
                }
                total /= wg_size;
                out_acc[group_id] = total;
            });
    });
    return 0;
}
