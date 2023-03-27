//==----------- element_wise_all_ops_bf16.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 8

#include "../element_wise_all_ops_bf16_impl.hpp"
