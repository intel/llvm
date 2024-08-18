//==-------joint_matrix_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix
// VNNI transform is not supported yet by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}

// XFAIL: cpu

// This tests support of row major layout for matrix B which does automatic VNNI
// transform. This is currently only available on AMX and XMX of PVC

#include "../common.hpp"
#include "../joint_matrix_rowmajorA_rowmajorB_impl.hpp"
