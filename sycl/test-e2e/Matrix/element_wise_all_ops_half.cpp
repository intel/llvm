//==----------- element_wise_all_ops_half.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-fp16
// REQUIRES: matrix
// REQUIRES: matrix-fp16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

constexpr size_t TN = 16;

#include "element_wise_all_ops_half_impl.hpp"
