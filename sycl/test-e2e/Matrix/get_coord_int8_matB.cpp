//==----------- get_coord_int8_matB.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// XFAIL: cpu, gpu-intel-dg2

#include "common.hpp"
#include "get_coord_int8_matB_impl.hpp"
