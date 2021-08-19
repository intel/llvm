// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//
//==-- sub-group-store-const-ref.cpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks that sub_group::store supports const reference.
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
using namespace sycl;

void test(ext::oneapi::sub_group sg, global_ptr<int> ptr) { sg.store(ptr, 1); }
