//===---joint_matrix_opt_kernel_feature.cpp - DPC++ joint_matrix-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that exception will be thrown in case matrix parameters are
// incompatible on the current device

#include "../Inputs/common.hpp"
#include "../Inputs/joint_matrix_opt_kernel_feature_impl.hpp"
