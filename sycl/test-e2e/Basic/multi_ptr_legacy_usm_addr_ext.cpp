// RUN: %clangxx -D__ENABLE_USM_ADDR_SPACE__ -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -D__ENABLE_USM_ADDR_SPACE__ -DRESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==-- multi_ptr_legacy_usm_addr_ext.cpp - SYCL multi_ptr legacy test ext --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr_legacy.hpp"
