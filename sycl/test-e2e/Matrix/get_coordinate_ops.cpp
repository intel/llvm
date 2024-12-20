//==----------- get_coordinate_ops.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-ext_intel_matrix

// XFAIL: !igc-dev
// XFAIL-TRACKER: GSD-6376
// REQUIRES-INTEL-DRIVER: lin: 30049

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#include "get_coordinate_ops_impl.hpp"
