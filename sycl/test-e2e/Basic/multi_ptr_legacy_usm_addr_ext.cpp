// REQUIRES: accelerator
// RUN: %{build} -Wno-error=deprecated-declarations -D__ENABLE_USM_ADDR_SPACE__ -fsycl-dead-args-optimization -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -Wno-error=deprecated-declarations -D__ENABLE_USM_ADDR_SPACE__ -DRESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR -fsycl-dead-args-optimization -o %t1.out
// RUN: %{run} %t1.out

//==-- multi_ptr_legacy_usm_addr_ext.cpp - SYCL multi_ptr legacy test ext --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr_legacy.hpp"
