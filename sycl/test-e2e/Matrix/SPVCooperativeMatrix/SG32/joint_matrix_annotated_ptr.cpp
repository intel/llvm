//==-------- joint_matrix_annotated_ptr.cpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out
// RUN: %{run} %t.out

// XFAIL: cpu

// Currently row major B fails when annotated_ptr is used
// XFAIL: gpu

#include "../../common.hpp"

#define SG_SZ 32
constexpr size_t TN = 16;

#include "../../joint_matrix_annotated_ptr_impl.hpp"
