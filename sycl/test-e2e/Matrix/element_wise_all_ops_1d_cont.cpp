//==-------- element_wise_all_ops_1d_cont.cpp - DPC++ joint_matrix ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix, gpu

// https://github.com/intel/llvm/issues/14826
// XFAIL: arch-intel_gpu_pvc

// RUN: %{build} -o %t.out
// RUN: env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out

#include "common.hpp"
#include "element_wise_all_ops_impl.hpp"
