//==------------ element_wise_all_ops.cpp  - DPC++ joint_matrix-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: ext_intel_matrix
// Test is flaky/timeouts on some variants of DG2 and temporary disabled. Needs
// to be investigated.
// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

#include "element_wise_all_ops_impl.hpp"
