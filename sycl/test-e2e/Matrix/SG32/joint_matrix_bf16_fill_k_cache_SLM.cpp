//==--- joint_matrix_bf16_fill_k_cache_SLM.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2
// REQUIRES: aspect-ext_intel_matrix, gpu

// RUN: %{build} -o %t_gpu_vnni.out %fp-model-precise -DSLM -DVNNI
// RUN: %{run} %t_gpu_vnni.out

// TODO: add row major compilation and run once Sub-group size 32
// support becomes available in IGC for row major

// -ffp-model=precise is added to not depend on compiler defaults.

#include "../common.hpp"
#define SG_SZ 32

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
