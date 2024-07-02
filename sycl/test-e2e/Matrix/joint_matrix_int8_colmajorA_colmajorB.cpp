//==----- joint_matrix_int8_colmajorA_colmajorB.cpp  - DPC++ joint_matrix---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This tests support of col major layout for matrix B which does transpose and
// then VNNI transform. This is currently only available on AMX

// XFAIL: gpu

#include "common.hpp"

constexpr size_t TN = 16;

#include "joint_matrix_int8_colmajorA_colmajorB_impl.hpp"
