// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify  %s -Xclang -verify-ignore-unexpected=note,warning
//=- implicit_conversion_error.cpp - Unintended implicit conversion check -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue q;
  cl::sycl::context cxt = q.get_context();
  cl::sycl::device dev = q.get_device();

  cl::sycl::context cxt2{dev};
  cl::sycl::context cxt3 = dev; // expected-error {{no viable conversion from 'cl::sycl::device' to 'cl::sycl::context'}}

  cl::sycl::queue q2{dev};
  cl::sycl::queue q3 = dev; // expected-error {{no viable conversion from 'cl::sycl::device' to 'cl::sycl::queue'}}
}
