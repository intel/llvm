//==-------- joint_matrix_annotated_ptr.cpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}

// XFAIL: cpu

#include "../common.hpp"

constexpr size_t TN = 16;

#include "../joint_matrix_annotated_ptr_impl.hpp"
