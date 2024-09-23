//==---------- joint_matrix_colA_rowB_colC.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// VNNI transform and sub-group size 32 are not supported yet on DG2 by IGC
// UNSUPPORTED: gpu-intel-dg2
// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

constexpr size_t TN = 16;

#include "joint_matrix_colA_rowB_colC_impl.hpp"
