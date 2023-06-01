// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    cl::sycl::queue sycl_queue;

    const int height = 8;
    const int width = 8;
    auto image_range = cl::sycl::range<2>(height, width);
    const int channels = 4;
    std::vector<float> in_data(height * width * channels, 0.5f);
    std::vector<float> out_data(height * width * channels, 0);

    cl::sycl::image<2> image_in(
        in_data.data(), cl::sycl::image_channel_order::rgba,
        cl::sycl::image_channel_type::fp32, image_range);
    cl::sycl::image<2> image_out(
        out_data.data(), cl::sycl::image_channel_order::rgba,
        cl::sycl::image_channel_type::fp32, image_range);

    auto work_range =
        cl::sycl::nd_range<2>(image_range, cl::sycl::range<2>(1, 1));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::accessor<cl::sycl::float4, 2, cl::sycl::access::mode::read,
                           cl::sycl::access::target::image>
            in_acc(image_in, cgh);
        cl::sycl::accessor<cl::sycl::float4, 2, cl::sycl::access::mode::write,
                           cl::sycl::access::target::image>
            out_acc(image_out, cgh);

        cl::sycl::sampler smpl(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::clamp,
            cl::sycl::filtering_mode::nearest);

        cgh.parallel_for<class image_copy>(
            work_range, [=](cl::sycl::nd_item<2> item_id) {
                auto coords = cl::sycl::int2(item_id.get_global_id(0),
                                             item_id.get_global_id(1));
                out_acc.write(coords, in_acc.read(coords, smpl));
            });
    });
    return 0;
}
