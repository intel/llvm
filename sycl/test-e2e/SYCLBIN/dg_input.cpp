//==----------- dg_input.cpp --- SYCLBIN extension tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// -- Test for using device globals in SYCLBIN.

// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287

// SYCLBIN currently only properly detects SPIR-V binaries.
// XFAIL: !target-spir
// XFAIL-TRACKER: CMPLRLLVM-68811

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/dg_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/dg.hpp"
