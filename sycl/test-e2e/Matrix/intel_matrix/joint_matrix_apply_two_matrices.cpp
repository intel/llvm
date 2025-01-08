//==------ joint_matrix_apply_two_matrices.cpp  - DPC++ joint_matrix--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} %fp-model-precise -o %t.out
// RUN: %{run} %t.out

#include "../Inputs/common.hpp"
#include "../Inputs/joint_matrix_apply_two_matrices_impl.hpp"
