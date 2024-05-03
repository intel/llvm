//== joint_matrix_bf16_fill_k_cache_init.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out -DINIT_LIST -ffp-model=precise
// RUN: %{run} %t.out

// -ffp-model=precise is added to not depend on compiler defaults.

#include "../common.hpp"
#include <cstddef>

#define SG_SZ 8
constexpr size_t TN = 8;

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
