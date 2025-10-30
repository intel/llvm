//==----------- dae_input.cpp --- SYCLBIN extension tests ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce input-state SYCLBIN files.

// -- Test for using a kernel from a SYCLBIN with a dead argument.

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/dae_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/dae.hpp"
