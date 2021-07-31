// REQUIRES: level_zero

// RUN: env SYCL_DEVICE_FILTER=level_zero sycl-ls | FileCheck %s --check-prefixes=CHECK-LEVELZERO

// CHECK-LEVELZERO-COUNT-1: [level_zero:{{.*}}:0]
// CHECK-LEVELZERO-NOT: [level_zero:{{.*}}:0]

//==-- sycl-ls-unique-device-id-level-zero.cpp - SYCL test for unique device id
//--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//