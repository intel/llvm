// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple /MD -o %t1.exe %s
// RUN: %CPU_RUN_PLACEHOLDER %t1.exe
// RUN: %GPU_RUN_PLACEHOLDER %t1.exe
// RUN: %ACC_RUN_PLACEHOLDER %t1.exe
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple /MDd -o %t2.exe %s
// RUN: %CPU_RUN_PLACEHOLDER %t2.exe
// RUN: %GPU_RUN_PLACEHOLDER %t2.exe
// RUN: %ACC_RUN_PLACEHOLDER %t2.exe
// REQUIRES: system-windows, cl_options
//==-------------- msvc_crt.cpp - SYCL MSVC CRT test -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MSVC provides two different incompatible variants of CRT: debug and release.
// This test checks if clang driver is able to handle this properly.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  int data[] = {0, 0, 0};

  {
    buffer<int, 1> b(data, range<1>(3), {property::buffer::use_host_ptr()});
    queue q;
    q.submit([&](handler &cgh) {
      auto B = b.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class test>(range<1>(3), [=](id<1> idx) { B[idx] = 1; });
    });
  }

  bool isSuccess = true;

  for (int i = 0; i < 3; i++)
    if (data[i] != 1)
      isSuccess = false;

  if (!isSuccess)
    return -1;

  return 0;
}
