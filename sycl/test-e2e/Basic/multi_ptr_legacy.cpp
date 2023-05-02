// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -DRESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==-------- multi_ptr_legacy.cpp - SYCL multi_ptr legacy test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr_legacy.hpp"
