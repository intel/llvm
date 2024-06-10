//==--- joint_matrix_int8_rowmajorA_rowmajorB.cpp  - DPC++ joint_matrix-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2
// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: gpu

#include "../common.hpp"

#define SG_SZ 32

#include "../joint_matrix_int8_rowmajorA_rowmajorB_impl.hpp"
