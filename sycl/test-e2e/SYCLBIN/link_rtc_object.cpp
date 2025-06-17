//==------------ link_rtc_object.cpp --- SYCLBIN extension tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// Fails for CUDA target due to new offload driver regression.
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18432

// -- Test for linking where one kernel is runtime-compiled and one is compiled
// -- to SYCLBIN.

// RUN: %clangxx %{sycl_target_opts} --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run}  %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link_rtc.hpp"
