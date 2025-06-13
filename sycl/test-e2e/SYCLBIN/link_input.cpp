//==-------------- link_input.cpp --- SYCLBIN extension tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_shared_allocations

// HIP and CUDA cannot answer kernel name queries on the binaries, so kernel
// names cannot be resolved for now.
// XFAIL: cuda || hip
// XFAIL-TRACKER: CMPLRLLVM-68469

// -- Test for linking two SYCLBIN kernel_bundle.

// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run}  %t.out %t.export.syclbin %t.import.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/link.hpp"
