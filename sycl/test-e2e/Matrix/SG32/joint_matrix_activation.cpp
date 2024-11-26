//==---------- joint_matrix_activation.cpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-INTENDED: SG size = 32 is not currently supported for SYCL Joint
// Matrix by IGC on DG2

// RUN: %{build} %fp-model-precise -o %t.out
// RUN: %{run} %t.out

// Currently, the outlining into an apply function triggers a bug in IGC
// XFAIL: gpu
// XFAIL-TRACKER: GSD-10373

#include "../common.hpp"

#define SG_SZ 32

#include "../joint_matrix_activation_impl.hpp"
