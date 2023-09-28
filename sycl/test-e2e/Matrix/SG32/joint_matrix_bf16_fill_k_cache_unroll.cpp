//==---joint_matrix_bf16_fill_k_cache_unroll.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %if gpu %{ %{build} -mllvm -inline-threshold=5000 -o %t_gpu.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -DMANUAL_UNROLL %}
// RUN: %if gpu %{ %{run} %t_gpu.out %}

// RUN: %if cpu %{ %{build} -mllvm -inline-threshold=5000 -o %t_cpu.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -DMANUAL_UNROLL -DtM=16 -DtK=32 -DNCACHE1=32 -DKCACHE1=32
// RUN: %if cpu %{ %{run} %t_cpu.out %}

// XFAIL: gpu

// -mllvm -inline-threshold added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently

#include <cstddef>

constexpr size_t SG_SZ = 32;
constexpr size_t TN = 16;

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
