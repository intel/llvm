//==-------SG32/joint_matrix_half_accumulator.cpp  - DPC++ joint_matrix ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-INTENDED: SG size = 32 is not supported for SYCL Joint Matrix on
// DG2
// UNSUPPORTED: cpu
// UNSUPPORTED-INTENDED: Different C and D types are not supported on AMX

// REQUIRES: target-spir
// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// XFAIL: gpu
// XFAIL-TRACKER: GSD-10112, GSD-4181

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}

#include "common.hpp"

#define SG_SZ 32

#include "joint_matrix_16bit_impl.hpp"

int main() {
  std::cout << "B row major:\n";
  test_all<half, float, layout::row_major, (size_t)1>();
  std::cout << "B packed:\n";
  test_all<half, float, layout::ext_intel_packed, (size_t)2>();
}
