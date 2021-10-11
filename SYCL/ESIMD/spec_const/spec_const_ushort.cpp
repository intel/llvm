//==--------------- spec_const_ushort.cpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -I%S/.. -DSYCL2020 %s -o %t.2020.out
// RUN: %GPU_RUN_PLACEHOLDER %t.2020.out
// UNSUPPORTED: cuda || hip

#include <cstdint>

#define DEF_VAL 0xcafe
#define REDEF_VAL 0xdeaf
#define STORE 2

using spec_const_t = uint16_t;
using container_t = uint16_t;

#ifndef SYCL2020
#include "Inputs/spec_const_common.hpp"
#else
#include "Inputs/spec-const-2020-common.hpp"
#endif
