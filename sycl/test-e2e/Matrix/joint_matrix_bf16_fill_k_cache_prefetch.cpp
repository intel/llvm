//==--- joint_matrix_bf16_fill_k_cache_prefetch.cpp  - DPC++ joint_matrix---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix
// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-INTENDED: prefetch is not supported on DG2

// RUN: %{build} -o %t_vnni.out -DPREFETCH -DVNNI %fp-model-precise
// RUN: %{run} %t_vnni.out

// RUN: %{build} -o %t.out -DPREFETCH %fp-model-precise
// RUN: %{run} %t.out

// -ffp-model=precise is added to not depend on compiler defaults.

#include "common.hpp"
#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
