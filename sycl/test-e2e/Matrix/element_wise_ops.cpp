//==----------- element_wise_ops.cpp  - DPC++ joint_matrix------------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix

// This test fails on LNL Windows machines with new driver versions.
// XFAIL: windows && intel_gpu_lnl_m
// XFAIL-TRACKER: CMPLRLLVM-72503

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#include "element_wise_ops_impl.hpp"
