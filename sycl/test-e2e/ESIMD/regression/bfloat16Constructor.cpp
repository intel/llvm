// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 || cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: gpu && !esimd_emulator
//==- bfloat16Constructor.cpp - Test to verify use of bfloat16 constructor -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is basic test to verify use of bfloat16 constructor in kernel.
// TODO: Enable the test once the GPU RT supporting the functionality reaches
// the CI

#include <iostream>
#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>

using namespace sycl;


int main() {
    constexpr unsigned Size = 32;
    constexpr unsigned VL = 32;
    constexpr unsigned GroupSize = 1;

    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto *C = malloc_shared<float>(Size * sizeof(float), dev, q.get_context());

    for (auto i = 0; i != Size; i++) {
        C[i] = 7;
    }

    nd_range<1> Range(range<1>(Size/VL), range<1>(GroupSize));

    auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Test>(
                Range, [=](nd_item<1> i)  SYCL_ESIMD_KERNEL {
                    using bf16 = sycl::ext::oneapi::bfloat16;
                    using namespace __ESIMD_NS;
                    using namespace __ESIMD_ENS;
                    simd<bf16, 32> data_bf16 = bf16(0);
                    simd<float, 32> data = data_bf16;
                    lsc_block_store<float, 32>(C, data);
                });
    });
    e.wait();
    bool Pass = true;
    for (auto i = 0; i != Size; i++) {
      if (C[i] != 0) {
        Pass = false;
      }
    }

    free(C, q);
    std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
    return 0;
}
