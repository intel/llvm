//==----------- element_wise_ops.cpp  - DPC++ joint_matrix------------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h
// XFAIL-TRACKER: CMPLRLLVM-66710

// XFAIL: linux && arch-intel_gpu_bmg_g21
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20190

#include "common.hpp"
#include "element_wise_ops_impl.hpp"
