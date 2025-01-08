//==---- element_wise_all_ops_scalar.cpp - DPC++ joint_matrix ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// REQUIRES-INTEL-DRIVER: lin: 30049

// RUN: %{build} -o %t.out
// RUN: env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out

#include "../Inputs/common.hpp"
#include "../Inputs/element_wise_all_ops_impl.hpp"
