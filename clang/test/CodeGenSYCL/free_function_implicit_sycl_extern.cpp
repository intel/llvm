//==---- free_function_implicit_sycl_extern.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -c -Xclang -verify %s

// expected-no-diagnostics

// This test checks that specifying SYCL_EXTERNAL is not necessary on a free
// function definition. The free function property implies SYCL_EXTERNAL.
// If this feature were not implemented correctly then each call to
// get_kernel_id would produce a diagnostic because the expected free function
// definition would be missing.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

using namespace sycl;

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel)) void ff_0(int *ptr,
                                                               int start,
                                                               int end) {}
// Overloaded free function definition.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void ff_1(int *ptr, int start, int end) {}

// Overloaded free function definition.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_1(int *ptr, int start) {}

// Templated free function definition.
template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_3(T *ptr, T start) {}

// Explicit instantiation with “int*”.
template void ff_3(int *ptr, int start);

void test(queue Queue) {
#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());

  kernel_id Kernel_id0 = ext::oneapi::experimental::get_kernel_id<ff_0>();

  kernel_id Kernel_id1 = ext::oneapi::experimental::get_kernel_id<(
      void (*)(int *, int, int))ff_1>();

  kernel_id Kernel_id2 =
      ext::oneapi::experimental::get_kernel_id<(void (*)(int *, int))ff_1>();

  kernel_id Kernel_id3 = ext::oneapi::experimental::get_kernel_id<(
      void (*)(int *, int))ff_3<int>>();
#endif
}
