//==----- joint_matrix_bfloat16_32x64x16.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix
// REQUIRES-INTEL-DRIVER: lin: 27868, win: 101.5181

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: cpu

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16
constexpr size_t TM = 32;
constexpr size_t TN = 64;
constexpr size_t TK = 16;

#include "joint_matrix_bfloat16_packedB_impl.hpp"
