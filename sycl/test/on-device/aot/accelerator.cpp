//==----- accelerator.cpp - AOT compilation for fpga devices using aoc  ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %S/Inputs/aot.cpp -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
