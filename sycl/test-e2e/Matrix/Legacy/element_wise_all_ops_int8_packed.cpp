//==------ element_wise_all_ops_int8_packed.cpp  - DPC++ joint_matrix-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1 -DSG_SZ=16
// RUN: %{run} %t.out
// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=1 -DSG_SZ=32
// RUN: %{run} %t.out

// XFAIL: *

#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;

#define TN 16

#include "element_wise_all_ops_int8_packed_impl.hpp"
