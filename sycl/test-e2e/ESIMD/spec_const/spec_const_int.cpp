//==--------------- spec_const_int.cpp  - DPC++ ESIMD on-device test ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out

#include <cstdint>

#define DEF_VAL 100500
#define REDEF_VAL -44556677
#define STORE 2

using spec_const_t = int32_t;
using container_t = int32_t;

#include "Inputs/spec-const-2020-common.hpp"
