//==-------- joint_matrix_prefetch.cpp  - DPC++ joint_matrix----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix
// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out
// RUN: %{run} %t.out

// XFAIL: cpu
// https://github.com/intel/llvm/issues/14826
// XFAIL: arch-intel_gpu_pvc && !igc-dev

#include "../common.hpp"

constexpr size_t TN = 16;
#include "../joint_matrix_prefetch_impl.hpp"
