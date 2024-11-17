//==------------- private_memory_pvc.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -I%S/.. -o %t.out
// RUN: %{run} %t.out

// The test verifies that basic ESIMD API works properly with
// private memory allocated on stack.

#define TEST_TFLOAT32
#include "private_memory.cpp"
