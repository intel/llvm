//==--- joint_matrix_bf16_fill_k_cache_SLM.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix, gpu

// RUN: %{build} -o %t_gpu_vnni.out %fp-model-precise -DSLM -DVNNI
// RUN: %{run} %t_gpu_vnni.out

// RUN: %{build} -o %t_gpu.out %fp-model-precise -DSLM
// RUN: %{run} %t_gpu.out

// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
