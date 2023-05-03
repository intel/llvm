// RUN: %clangxx -D__ENABLE_USM_ADDR_SPACE__ -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==----- multi_ptr_usm_addr_ext.cpp - SYCL multi_ptr test with usm ext ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr.hpp"
