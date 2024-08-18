//==-- joint_matrix_bfloat16_colmajorA_colmajorB.cpp  - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out
// RUN: %{run} %t.out

// This tests support of col major layout for matrix B which does transpose and
// then VNNI transform. This is currently only available on AMX

// XFAIL: gpu

#include "../../common.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define SG_SZ 32
constexpr size_t TN = 16;

#include "../../joint_matrix_bfloat16_colmajorA_colmajorB_impl.hpp"
