//==---------- sycl-post-tool test for bfloat16 devicelib support  ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: sycl-post-link -split=auto -emit-param-info -symbols -emit-imported-symbols -properties -o %t.txt %t.bc
// RUN: cat %t_0.prop | FileCheck %s -check-prefixes=CHECK-BF16
// UNSUPPORTED: target-nvidia || target-amd

// sycl-post-link tool will analyze device code to check whether sycl bfloat16
// device library is used. If yes, the used functions will be added to imported
// symbols list.
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

using BFP = sycl::ext::oneapi::bfloat16;

int main() {
  BFP bf16_array[3];
  float fp32_array[3] = {7.0f, 8.5f, 0.5f};

  sycl::queue deviceQueue;
  {
    buffer<float, 1> fp32_buffer{fp32_array, 3};
    buffer<BFP, 1> bf16_buffer{bf16_array, 3};
    deviceQueue
        .submit([&](handler &cgh) {
          auto fp32_acc = fp32_buffer.template get_access<sycl_read>(cgh);
          auto bf16_acc = bf16_buffer.template get_access<sycl_write>(cgh);
          cgh.single_task([=]() {
            bf16_acc[0] = BFP{fp32_acc[0]};
            bf16_acc[1] = BFP{fp32_acc[1]};
            bf16_acc[2] = BFP{fp32_acc[2]};
          });
        })
        .wait();
  }
  std::cout << bf16_array[0] << " " << bf16_array[1] << " " << bf16_array[2]
            << std::endl;
  return 0;
}

// CHECK-BF16: [SYCL/imported symbols]
// CHECK-BF16-NEXT: __devicelib_ConvertFToBF16INTEL
