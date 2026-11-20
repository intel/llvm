//==--joint_matrix_bf16_fill_k_cache_unroll_init.cpp  - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir
// REQUIRES: aspect-ext_intel_matrix, gpu

// The default iteration count does not finish on the CRI simulator,
// CMPLRLLVM-75924, so run fewer iterations there.
// RUN: %{build} -mllvm -inline-threshold=2000 %fp-model-precise -o %t_gpu.out -DINIT_LIST -DMANUAL_UNROLL -DVNNI %if arch-intel_gpu_cri %{ -DTESTITERATIONS=11 %}
// RUN: %{run} %t_gpu.out

// -mllvm -inline-threshold=2000 added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently
// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
