//===-lsc_block_load_store_stateless_64_flag - DPC++ ESIMD on-device test-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out -fsycl-esimd-force-stateless-mem
// RUN: %{run} %t.out

#define TEST_FLAG 1

#include "lsc_block_load_store_stateless_64.cpp"
