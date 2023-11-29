//==--joint_matrix_bfloat16_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This tests support of row major layout for matrix B which does automatic VNNI
// transform. This is currently only available on AMX and XMX of PVC

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16
constexpr size_t TN = 16;

#include "joint_matrix_bfloat16_rowmajorA_rowmajorB_impl.hpp"
