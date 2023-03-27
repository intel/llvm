//==--joint_matrix_bfloat16_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This tests support of row major layout for matrix B which does automatic VNNI
// transform. This is currently only available on AMX

// XFAIL: gpu

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define SG_SZ 16

#include "joint_matrix_bfloat16_rowmajorA_rowmajorB_impl.hpp"
