//==------------ link_rtc_input.cpp --- SYCLBIN extension tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking where one kernel is runtime-compiled and one is compiled
// -- to SYCLBIN.

// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/link_rtc.hpp"
