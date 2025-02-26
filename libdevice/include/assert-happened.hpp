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

#if defined(__SPIR__) || defined(__SPIRV__)

// NOTE Layout of this structure should be aligned with the one in
// sycl/include/sycl/detail/assert_happened.hpp
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

#ifndef SPIR_GLOBAL_VAR
#ifdef __SYCL_DEVICE_ONLY__
#define SPIR_GLOBAL_VAR __attribute__((sycl_global_var))
#else
#warning "SPIR_GLOBAL_VAR not defined in host mode. Defining as empty macro."
#define SPIR_GLOBAL_VAR
#endif
#endif

#define __SYCL_GLOBAL__ __attribute__((opencl_global))

// declaration
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ AssertHappened SPIR_AssertHappenedMem;

#endif
