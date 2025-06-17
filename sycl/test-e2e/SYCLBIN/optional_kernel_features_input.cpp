//==--- optional_kernel_features_input.cpp --- SYCLBIN extension tests -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// -- Test for compiling and loading a kernel bundle with a SYCLBIN containing
//    the use of optional kernel features.

// Fails for CUDA target due to new offload driver regression.
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18432

// RUN: %clangxx %{sycl_target_opts} --offload-new-driver -fsyclbin=input %S/Inputs/optional_kernel_features.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/optional_kernel_features.hpp"
