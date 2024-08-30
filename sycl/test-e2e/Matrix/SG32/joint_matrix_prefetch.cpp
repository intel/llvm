//==-------- joint_matrix_prefetch.cpp  - DPC++ joint_matrix----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943
// REQUIRES: aspect-ext_intel_matrix
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: gpu

// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2

#include "../common.hpp"

#define SG_SZ 32
constexpr size_t TN = 16;
#include "../joint_matrix_prefetch_impl.hpp"
