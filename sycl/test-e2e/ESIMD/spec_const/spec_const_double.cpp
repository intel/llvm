//==--------------- spec_const_double.cpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-fp64
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out

#include <cstdint>

#define DEF_VAL 9.1029384756e+11
#define REDEF_VAL -1.4432211654e-10
#define STORE 1

using spec_const_t = double;
using container_t = double;

#include "Inputs/spec-const-2020-common.hpp"
