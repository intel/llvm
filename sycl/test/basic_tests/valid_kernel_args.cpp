//==----------- valid_kernel_args.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The test checks that the types can be used to pass kernel parameters by value
// RUN: %clangxx -fsycl -fsyntax-only %s -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning

// Check that the test can be compiled with device compiler as well.
// RUN: %clangxx -fsycl-device-only -fsyntax-only %s -Wno-sycl-strict

#include <CL/sycl.hpp>

struct SomeStructure {
  char a;
  float b;
  union {
    int x;
    double y;
  } v;
};

#define CHECK_PASSING_TO_KERNEL_BY_VALUE(Type)                                 \
  static_assert(std::is_standard_layout<Type>::value,                          \
                "Is not standard layouti type.");                     \
  static_assert(std::is_trivially_copyable<Type>::value,                       \
                "Is not trivially copyable type.");

#ifdef __SYCL_DEVICE_ONLY__
CHECK_PASSING_TO_KERNEL_BY_VALUE(int)
CHECK_PASSING_TO_KERNEL_BY_VALUE(cl::sycl::cl_uchar4)
CHECK_PASSING_TO_KERNEL_BY_VALUE(SomeStructure)
#endif
