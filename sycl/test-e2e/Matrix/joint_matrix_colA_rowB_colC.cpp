//==---------- joint_matrix_colA_rowB_colC.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix

// XFAIL: run-mode && gpu-intel-dg2
// XFAIL-TRACKER: GSD-5768

// The default size hangs on the CRI simulator, CMPLRLLVM-75924, so build a
// smaller one there. Restore the default once CRI hardware is available.
// RUN: %if arch-intel_gpu_cri %{ %{build} -DMATRIX_SIZE=256 -o %t.out %} %else %{ %{build} -o %t.out %}
// RUN: %{run} %t.out

#include "common.hpp"
#include "joint_matrix_colA_rowB_colC_impl.hpp"
