//==------- memcpy2d_dhost_to_dhost.cpp - 2D memcpy from host to host ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::DirectHost, Alloc::DirectHost>(); }
