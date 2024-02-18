// RUN: %{build} -fsycl-dead-args-optimization -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DRESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR -fsycl-dead-args-optimization -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t1.out
// RUN: %{run} %t1.out

//==-------- multi_ptr_legacy.cpp - SYCL multi_ptr legacy test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "multi_ptr_legacy.hpp"
