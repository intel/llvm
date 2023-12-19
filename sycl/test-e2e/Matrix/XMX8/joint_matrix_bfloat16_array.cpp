//==-------- joint_matrix_bfloat16_array.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../common.hpp"

#define SG_SZ 8
static constexpr int TN = 8;

#include "../joint_matrix_bfloat16_array_impl.hpp"
