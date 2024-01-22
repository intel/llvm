//==---joint_matrix_bf16_fill_k_cache_unroll.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -mllvm -inline-threshold=5000 -ffp-model=precise -o %t_gpu.out -DMANUAL_UNROLL
// RUN: %if gpu %{ %{run} %t_gpu.out %}

// RUN: %{build} -mllvm -inline-threshold=5000 -ffp-model=precise -o %t_cpu.out -DMANUAL_UNROLL -DtM=16 -DtK=32 -DNCACHE1=32 -DKCACHE1=32
// RUN: %if cpu %{ %{run} %t_cpu.out %}

// -mllvm -inline-threshold added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently
// -ffp-model=precise is added to not depend on compiler defaults.

#include "../common.hpp"
#include <cstddef>

constexpr size_t SG_SZ = 32;
constexpr size_t TN = 16;

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
