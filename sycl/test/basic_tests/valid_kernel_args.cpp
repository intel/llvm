//==----------- valid_kernel_args.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The test checks that the types can be used to pass kernel parameters by value
// RUN: %clangxx -fsycl -fsyntax-only %s -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fsyntax-only -fpreview-breaking-changes %s -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning %}

// Check that the test can be compiled with device compiler as well.
// RUN: %clangxx -fsycl-device-only -fsyntax-only %s -Wno-sycl-strict
// RUN: %if preview-breaking-changes-supported %{%clangxx -fsycl-device-only -fsyntax-only -fpreview-breaking-changes %s -Wno-sycl-strict%}

#include <sycl/sycl.hpp>

struct SomeStructure {
  char a;
  float b;
  union {
    int x;
    double y;
  } v;
};

struct SomeMarrayStructure {
  sycl::marray<double, 5> points;
};

template <typename T> void check() {
  static_assert(std::is_standard_layout<T>::value,
                "Is not standard layouti type.");
  static_assert(std::is_trivially_copyable<T>::value,
                "Is not trivially copyable type.");
}

SYCL_EXTERNAL void foo() {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES

  check<int>();
  check<sycl::vec<sycl::opencl::cl_uchar, 4>>();
  check<SomeStructure>();
  check<sycl::int4>();
  check<sycl::long16>();

#else // __INTEL_PREVIEW_BREAKING_CHANGES

#ifdef __SYCL_DEVICE_ONLY__
  check<int>();
  check<sycl::vec<sycl::opencl::cl_uchar, 4>>();
  check<SomeStructure>();
#endif // __SYCL_DEVICE_ONLY
#endif //__INTEL_PREVIEW_BREAKING_CHANGES

  check<SomeMarrayStructure>();
}
