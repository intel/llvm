//==--------- basic_object.cpp --- SYCLBIN extension tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// -- Basic test for compiling and loading a SYCLBIN kernel_bundle in object
// -- state.

// SYCLBIN currently only properly detects SPIR-V binaries.
// XFAIL: !target-spir
// XFAIL-TRACKER: CMPLRLLVM-68811

// RUN: %clangxx --offload-new-driver -fsyclbin=object %{sycl_target_opts} %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/basic.hpp"
