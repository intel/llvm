//==----- fpga_device_link_diag.cpp - SYCL FPGA linking diagnostic test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator

// RUN: %clangxx -fsycl -fintelfpga %s -c -o test_obj.o
// RUN: touch dummy.cpp
// RUN: %clangxx -fsycl -fintelfpga dummy.cpp -c
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image dummy.o -o dummy_arch.a
// RUN: not %clangxx -fsycl -fintelfpga test_obj.o dummy_arch.a 2>&1 | FileCheck %s --check-prefix=CHK-FPGA-LINK-DIAG
// CHK-FPGA-LINK-DIAG: note: diagnostic msg: The FPGA image does not include all device kernels from test_obj.o. Please re-generate the image

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

void foo() {
  kernel_single_task<class kernel>([]() {});
}
