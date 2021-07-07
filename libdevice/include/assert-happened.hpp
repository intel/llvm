//==-- assert-happened.hpp - Structure and declaration for assert support --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

#ifdef __SPIR__

struct AssertHappened {
  int Flag = 0;
  char Expr[256 + 1] = "";
  char File[256 + 1] = "";
  char Func[128 + 1] = "";

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;
};

#ifndef __SYCL_GLOBAL_VAR__
#define __SYCL_GLOBAL_VAR__ __attribute__((sycl_global_var))
#endif

#define __SYCL_GLOBAL__ __attribute__((opencl_global))

// declaration
extern __SYCL_GLOBAL_VAR__ __SYCL_GLOBAL__ AssertHappened
    __SYCL_AssertHappenedMem;

#endif
