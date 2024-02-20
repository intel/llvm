//==---------------- BitonicSortKv2.cpp  - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Use -O2 to avoid huge stack usage under -O0.
// RUN: %{build} -O2 -o %t.out
// RUN: %{run} %t.out

#include "BitonicSortKv2.hpp"
