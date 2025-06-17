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

// RUN: %clangxx %{sycl_target_opts} --offload-new-driver -fsyclbin=object %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/basic.hpp"
