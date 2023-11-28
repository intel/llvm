//==------ element_wise_all_ops_int8_packed.cpp  - DPC++ joint_matrix-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8
// REQUIRES: TEMPORARY_DISBLED

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test stores the matrix B that is VNNIed (packed).

#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 8
constexpr size_t TN = 8;

#include "../element_wise_all_ops_int8_packed_impl.hpp"
