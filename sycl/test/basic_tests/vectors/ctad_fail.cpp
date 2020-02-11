// RUN: %clangxx -std=c++17 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected %s
//==--------------- ctad.cpp - SYCL vector CTAD fail test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main() {
  sycl::vec v(1, .1); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'vec'}}
}
