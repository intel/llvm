//==--------------- spec_const_char.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip

#include <cstdint>

#define DEF_VAL -22
#define REDEF_VAL 33
#define STORE 2

using spec_const_t = int8_t;
using container_t = int8_t;

#include "Inputs/spec_const_common.hpp"
