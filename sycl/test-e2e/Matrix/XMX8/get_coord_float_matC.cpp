//==----------- get_coord_float_matC.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix-xmx8
// REQUIRES-INTEL-DRIVER: lin: 28267

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// XFAIL: cpu

#include "../common.hpp"
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t SG_SZ = 8;
constexpr size_t TN = 8;

#include "../get_coord_float_matC_impl.hpp"
