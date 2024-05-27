//==-------- joint_matrix_out_bounds.cpp  - DPC++ joint_matrix--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL:*

#include "common.hpp"

constexpr size_t TN = 16;
constexpr size_t MATRIX_K = 1024 + 24;

#include "joint_matrix_out_bounds_impl.hpp"
