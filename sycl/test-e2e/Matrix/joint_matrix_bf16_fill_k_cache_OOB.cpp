//==--- joint_matrix_bf16_fill_k_cache_OOB.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix, gpu
// UNSUPPORTED: gpu-intel-dg2

// https://github.com/intel/llvm/issues/14826
// XFAIL: arch-intel_gpu_pvc && igc-dev

// RUN: %{build} -o %t_gpu_vnni.out -ffp-model=precise -DOOB -DVNNI
// RUN: %{run} %t_gpu_vnni.out

// RUN: %{build} -o %t_gpu.out -ffp-model=precise -DOOB
// RUN: %{run} %t_gpu.out

// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
