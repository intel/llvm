//==----------- element_wise_all_sizes.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This is a version of the test with disabled device code
// split to test against fixed bug in IGC
// RUN: %{build} -fsycl-device-code-split=off -o %t_split.out
// RUN: %if gpu-intel-dg2 %{ %{run} %t_split.out %}

#include "common.hpp"
#include "element_wise_all_sizes_impl.hpp"
