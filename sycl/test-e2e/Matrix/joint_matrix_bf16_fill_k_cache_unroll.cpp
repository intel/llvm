//==---joint_matrix_bf16_fill_k_cache_unroll.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -mllvm -inline-threshold=2000 -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -DMANUAL_UNROLL
// RUN: %{run} %t.out

// Optimizations:
// cache tiling of i and j
// cache tiling on k as well (so no reordering is needed)
// data reuse of A and B in physical layer

// -mllvm -inline-threshold=2000 added as a workaround,
// since IGC doesn't support some variants of IR for Joint Matrix currently

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sycl/sycl.hpp>

#include <err.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define SG_SZ 16

#include "joint_matrix_bf16_fill_k_cache_impl.hpp"
