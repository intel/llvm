//==--------------- spec_const_bool.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out

#include <cstdint>

#define DEF_VAL true
#define REDEF_VAL false
#define STORE 0

// In this case container type is set to unsigned char to be able to use
// esimd memory interfaces to pollute container.
using spec_const_t = bool;
using container_t = uint8_t;

#include "Inputs/spec-const-2020-common.hpp"
