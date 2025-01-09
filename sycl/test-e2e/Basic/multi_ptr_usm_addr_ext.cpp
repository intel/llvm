// REQUIRES: accelerator
// UNSUPPORTED-TRIPLES: amdgcn-amd-amdhsa
// RUN: %{build} -D__ENABLE_USM_ADDR_SPACE__ -fsycl-dead-args-optimization -o %t.out
// RUN: %{run} %t.out

//==----- multi_ptr_usm_addr_ext.cpp - SYCL multi_ptr test with usm ext ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr.hpp"
