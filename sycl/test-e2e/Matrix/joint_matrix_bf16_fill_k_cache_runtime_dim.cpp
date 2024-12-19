//==--- joint_matrix_bf16_fill_k_cache_runtime_dim.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t_runtime_dim_vnni.out %fp-model-precise -DRUNTIME_DIM -DVNNI
// RUN: %{run} %t_runtime_dim_vnni.out 256

// -ffp-model=precise is added to not depend on compiler defaults.

// Waiting for the commit in IGC to be pulled into the driver to resolve the
// test.
// XFAIL: (!igc-dev || gpu-intel-dg2) && run-mode
// XFAIL-TRACKER: CMPLRLLVM-63710

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
