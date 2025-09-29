//==---------- dg_executable.cpp --- SYCLBIN extension tests ---------------==//
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

// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19533

// RUN: %clangxx --offload-new-driver -fsyclbin=executable %{sycl_target_opts} %{syclbin_exec_opts} %S/Inputs/dg_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/dg.hpp"
