//==-------- joint_matrix_bfloat16_array.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#define SG_SZ 16
static constexpr int TN = 16;

#include "joint_matrix_bfloat16_array_impl.hpp"
