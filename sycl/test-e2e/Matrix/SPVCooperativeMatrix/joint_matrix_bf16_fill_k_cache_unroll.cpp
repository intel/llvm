//==---joint_matrix_bf16_fill_k_cache_unroll.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -mllvm -inline-threshold=2000 -ffp-model=precise -o %t.out -DMANUAL_UNROLL -DVNNI
// RUN: %{run} %t.out

// XFAIL: cpu

// -mllvm -inline-threshold=2000 added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently
// -ffp-model=precise is added to not depend on compiler defaults.

#include "../common.hpp"
#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
