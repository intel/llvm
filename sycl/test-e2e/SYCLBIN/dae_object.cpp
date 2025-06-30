//==--------- basic_object.cpp --- SYCLBIN extension tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// -- Test for using a kernel from a SYCLBIN with a dead argument.

// Due to the regression in https://github.com/intel/llvm/issues/18432 it will
// fail to build the SYCLBIN with nvptx targets. Once this is fixed,
// %{sycl_target_opts} should be added to the SYCLBIN generation run-line.
// REQUIRES: target-spir

// RUN: %clangxx --offload-new-driver -fsyclbin=object %S/Inputs/dae_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/dae.hpp"
