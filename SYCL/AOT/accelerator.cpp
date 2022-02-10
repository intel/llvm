//=-- accelerator.cpp - compilation for fpga emulator dev using opencl-aot --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: opencl-aot, accelerator

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %S/Inputs/aot.cpp -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
