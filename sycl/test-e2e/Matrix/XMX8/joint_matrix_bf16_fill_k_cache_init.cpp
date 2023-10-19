//== joint_matrix_bf16_fill_k_cache_init.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -DINIT_LIST
// RUN: %{run} %t.out

#include "../common.hpp"
#include <cstddef>

#define SG_SZ 8
constexpr size_t TN = 8;

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
