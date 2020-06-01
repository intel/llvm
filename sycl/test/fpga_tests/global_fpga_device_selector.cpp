//==--- global_fpga_device_selector.cpp - SYCL FPGA device selector test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator

// RUN: %clangxx -fsycl -fintelfpga -std=c++17 %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

// Check that FPGA emulator device is found if we try to initialize inline global
// variable using fpga_emulator_selector parameter.

inline cl::sycl::queue fpga_emu_queue_inlined{
    cl::sycl::intel::fpga_emulator_selector{}};

int main () {
    return 0;
}

