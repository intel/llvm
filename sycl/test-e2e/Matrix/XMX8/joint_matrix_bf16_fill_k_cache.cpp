//==--- joint_matrix_bf16_fill_k_cache.cpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out

// Optimizations:
// cache tiling of i and j
// cache tiling on k as well (so no reordering is needed)
// data reuse of A and B in physical layer

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

#define SG_SZ 8

#include "../joint_matrix_bf16_fill_k_cache_impl.hpp"
