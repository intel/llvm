//==--joint_matrix_bf16_fill_k_cache_unroll_init.cpp  - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2
// REQUIRES: aspect-ext_intel_matrix, gpu
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -mllvm -inline-threshold=5000 -ffp-model=precise -o %t_gpu.out -DINIT_LIST -DMANUAL_UNROLL
// RUN: %{run} %t_gpu.out

// -mllvm -inline-threshold added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently
// -ffp-model=precise is added to not depend on compiler defaults.

#include "../common.hpp"

#define SG_SZ 32

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
