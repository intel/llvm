//==---- free_function_errors.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx  -fsyntax-only -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <array>
#include <sycl/sycl.hpp>

using namespace sycl;

struct S {
  int i;
  float f;
};

union U {
  int i;
  float f;
};

using accType = accessor<int, 1, access::mode::read_write>;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff(struct S s) {}

// expected-error@+3 {{'union U' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff(union U u) {}

// expected-error@+3 {{'accType' (aka 'accessor<int, 1, access::mode::read_write>') cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff(accType acc) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff(std::array<int, 10> a) {}

// expected-error@+3 {{'int &' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff(int &ip) {}
