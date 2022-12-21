//==------ element_wise_all_ops_int8_packed.cpp  - DPC++ joint_matrix-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test stores the matrix B that is VNNIed (packed) in a row major fashion.
// This is expected to fail on the GPU because the implementation does not
// support automatic transformation YET, in this case: VNNI to row major in the
// store.

// XFAIL: gpu

#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

#include "element_wise_all_ops_int8_packed_impl.hpp"
