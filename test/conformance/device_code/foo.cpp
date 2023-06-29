// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <CL/sycl.hpp>

int main() {
    cl::sycl::queue deviceQueue;
    cl::sycl::range<1> numOfItems{1};

    deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto kern = [=](cl::sycl::id<1>) {};
        cgh.parallel_for<class Foo>(numOfItems, kern);
    });

    return 0;
}
