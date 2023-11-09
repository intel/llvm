//==----------- element_wise_abc.cpp  - DPC++ joint_matrix------------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cstddef>

#define SG_SZ 16
constexpr size_t TN = 16;

#include "element_wise_abc_impl.hpp"
