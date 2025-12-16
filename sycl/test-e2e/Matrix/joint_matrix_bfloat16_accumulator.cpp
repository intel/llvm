//==--- joint_matrix_bfloat16_accumulator.cpp  - DPC++ joint_matrix-- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: cpu
// UNSUPPORTED-INTENDED: Different C and D types are not supported on AMX

// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// XFAIL: gpu-intel-dg2
// XFAIL-TRACKER: GSD-10112

// XFAIL: linux && arch-intel_gpu_bmg_g21 && !igc-dev
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20594

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}

#include "common.hpp"

#include "joint_matrix_16bit_impl.hpp"

int main() {
  std::cout << "B row major:\n";
  test_all<bfloat16, float, layout::row_major, (size_t)1>();
  std::cout << "B packed:\n";
  test_all<bfloat16, float, layout::ext_intel_packed, (size_t)2>();
}
