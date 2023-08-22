//==---------- joint_matrix_colA_rowB_colC.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out

// XFAIL:gpu

#include "common.hpp"

constexpr size_t SG_SZ = 16;
constexpr size_t TN = 16;

#include "joint_matrix_colA_rowB_colC_impl.hpp"
