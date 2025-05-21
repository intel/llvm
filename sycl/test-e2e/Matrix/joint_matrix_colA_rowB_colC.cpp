//==---------- joint_matrix_colA_rowB_colC.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir || target-native_cpu

// REQUIRES: aspect-ext_intel_matrix

// XFAIL: run-mode && gpu-intel-dg2
// XFAIL-TRACKER: GSD-5768

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#include "joint_matrix_colA_rowB_colC_impl.hpp"
