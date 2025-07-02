//==-------------- link_input.cpp --- SYCLBIN extension tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking two SYCLBIN kernel_bundle.

// Due to the regression in https://github.com/intel/llvm/issues/18432 it will
// fail to build the SYCLBIN with nvptx targets. Once this is fixed,
// %{sycl_target_opts} should be added to the SYCLBIN generation run-line.
// REQUIRES: target-spir

// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run}  %t.out %t.export.syclbin %t.import.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link.hpp"
