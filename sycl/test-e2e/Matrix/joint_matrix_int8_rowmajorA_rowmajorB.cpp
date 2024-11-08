//==----- joint_matrix_int8_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Run these 2 tests on PVC only for now. Check can be updated to "gpu",
// when newer IGC is used in intel/llvm pre-checkin testing on Intel Arc
// RUN: %if arch-intel_gpu_pvc %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}
// RUN: %if arch-intel_gpu_pvc %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}

#include "common.hpp"
#include "joint_matrix_int8_rowmajorA_rowmajorB_impl.hpp"
