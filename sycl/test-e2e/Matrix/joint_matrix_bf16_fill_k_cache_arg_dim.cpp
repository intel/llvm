//==--- joint_matrix_bf16_fill_k_cache_arg_dim.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: aspect-ext_intel_matrix isn't currently supported for
// other triples
// XFAIL: run-mode && gpu
// XFAIL-TRACKER: CMPLRLLVM-66371

// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t_arg_dim_vnni.out %fp-model-precise -DARG_DIM -DVNNI
// RUN: %{run} %t_arg_dim_vnni.out

// -ffp-model=precise is added to not depend on compiler defaults.

// Waiting for the commit in IGC to be pulled into the driver to resolve the
// test.
// XFAIL: gpu-intel-dg2 && run-mode
// XFAIL-TRACKER: GSD-10510
// XFAIL: arch-intel_gpu_bmg_g21
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16922

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
