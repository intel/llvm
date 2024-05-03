//==------ joint_matrix_apply_two_matrices.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -ffp-model=precise -o %t.out
// RUN: %{run} %t.out

#include "../common.hpp"

#define SG_SZ 32

#include "../joint_matrix_apply_two_matrices_impl.hpp"
