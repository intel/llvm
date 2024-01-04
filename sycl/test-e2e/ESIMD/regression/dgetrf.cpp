//==-------------- dgetrf.cpp  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out 3 2 1
//
// This test checks the correctness of ESIMD program for batched LU
// decomposition without pivoting. The program contains multiple branches
// corresponding to LU input sizes; all internal functions are inlined.
//

#include "Inputs/dgetrf.hpp"
