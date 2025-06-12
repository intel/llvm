//==--------- basic_input.cpp --- SYCLBIN extension tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// HIP and CUDA cannot answer kernel name queries on the binaries, so kernel
// names cannot be resolved for now.
// XFAIL: cuda || hip
// XFAIL-TRACKER: CMPLRLLVM-68469

// -- Basic test for compiling and loading a SYCLBIN kernel_bundle in input
// -- state.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/basic.hpp"
