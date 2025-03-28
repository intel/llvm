//==-------- joint_matrix_annotated_ptr.cpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: aspect-ext_intel_matrix isn't currently supported for
// other triples

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-INTENDED: SG size = 32 is not currently supported for SYCL Joint
// Matrix by IGC on DG2

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

#define SG_SZ 32
constexpr size_t TN = 16;

#include "joint_matrix_annotated_ptr_impl.hpp"
