//==--- joint_matrix_bf16_fill_k_cache.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t_gpu.out -ffp-model=precise
// RUN: %if gpu %{ %{run} %t_gpu.out %}

// RUN: %{build}  -ffp-model=precise -o %t_cpu.out -DtM=16 -DtK=32 -DNCACHE1=32 -DKCACHE1=32
// RUN: %if cpu %{ %{run} %t_cpu.out %}

// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include <cstddef>

#define SG_SZ 16
constexpr size_t TN = 16;

#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
