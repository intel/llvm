//==- joint_matrix_bf16_rowmajorB_pair_load_store.cpp - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: cpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../../include/common.hpp"

#define SG_SZ 32

#include "../../include/joint_matrix_bf16_rowmajorB_pair_load_store_impl.hpp"
