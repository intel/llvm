//==--------joint_matrix_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943
// VNNI transform and sub-group size 32 are not supported yet on DG2 by IGC
// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This tests support of row major layout for matrix B which does automatic VNNI
// transform. This is currently only available on AMX

// XFAIL: gpu

#include "../common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 32

#include "../joint_matrix_rowmajorA_rowmajorB_impl.hpp"
