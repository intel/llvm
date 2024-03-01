//==--joint_matrix_bf16_fill_k_cache_unroll_init.cpp  - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix, gpu

// RUN: %{build} -mllvm -inline-threshold=2000 -ffp-model=precise -o %t_gpu.out -DINIT_LIST -DMANUAL_UNROLL
// RUN: %{run} %t_gpu.out

// -mllvm -inline-threshold=2000 added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently
// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include <cstddef>

#define SG_SZ 16
constexpr size_t TN = 16;

#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
