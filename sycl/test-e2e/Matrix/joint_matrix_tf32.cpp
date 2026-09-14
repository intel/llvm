//==---------------- joint_matrix_tf32.cpp  - DPC++ joint_matrix------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-L0-DRIVER: 27501
// REQUIRES-INTEL-WINDOWS-DRIVER: 101.4943

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

constexpr size_t TN = 16;

#include "joint_matrix_tf32_impl.hpp"
