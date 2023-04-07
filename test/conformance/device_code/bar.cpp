// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>

int main() {
    cl::sycl::queue deviceQueue;
    cl::sycl::range<1> numOfItems{1};

    deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto kern = [=](cl::sycl::id<1>) {};
        cgh.parallel_for<class Bar>(numOfItems, kern);
    });

    return 0;
}
