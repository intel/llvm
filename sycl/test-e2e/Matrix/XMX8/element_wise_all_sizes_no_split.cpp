//==-------- element_wise_all_sizes_no_split.cpp  - DPC++ joint_matrix------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a version of element_wise_all_sizes test with disabled device code
// split to test against fixed bug in IGC

// REQUIRES: matrix-xmx8
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -fsycl-device-code-split=off -o %t.out
// RUN: %{run} %t.out

#include "../common.hpp"

#define SG_SZ 8
constexpr size_t TN = 8;

#include "../element_wise_all_sizes_impl.hpp"
