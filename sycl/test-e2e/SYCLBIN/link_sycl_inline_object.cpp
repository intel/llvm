//==-------- link_sycl_inline_object.cpp --- SYCLBIN extension tests
//--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking between inline SYCL code and SYCLBIN code.

// ptxas currently fails to compile images with unresolved symbols. Disable for
// other targets than SPIR-V until this has been resolved. (CMPLRLLVM-68810)
// Note: %{sycl_target_opts} should be added to the SYCLBIN compilation lines
// once fixed.
// REQUIRES: target-spir

// XFAIL: opencl && cpu
// XFAIL-TRACKER: CMPLRLLVM-68800

// XFAIL: linux && arch-intel_gpu_bmg_g21
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19258

// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies -Xclang -fsycl-allow-func-ptr %S/Inputs/link_sycl_inline.cpp -o %t.syclbin
// RUN: %{build} -fsycl-allow-device-image-dependencies -o %t.out
// RUN: %{l0_leak_check} %{run}  %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link_sycl_inline.hpp"
