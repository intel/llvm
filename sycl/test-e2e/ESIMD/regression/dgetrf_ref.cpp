//==-------------- dgetrf_ref.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -Wno-error=format -DUSE_REF -I%S/.. -o %t.ref.out
// RUN: %{run} %t.ref.out 3 2 1
//
// This test checks the correctness of ESIMD program for batched LU
// decomposition without pivoting. The program contains multiple branches
// corresponding to LU input sizes; all internal functions are inlined.
//

#include "Inputs/dgetrf.hpp"
