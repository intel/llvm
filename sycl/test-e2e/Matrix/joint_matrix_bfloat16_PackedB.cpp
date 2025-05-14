//==-------- joint_matrix_bfloat16_PackedB.cpp  - DPC++ joint_matrix----------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2

// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: aspect-ext_intel_matrix isn't currently supported for
// other triples

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}

#include "common.hpp"
#include "joint_matrix_bfloat16_PackedB_impl.hpp"
