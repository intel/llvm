// RUN: %{run-unfiltered-devices} sycl-ls --verbose | grep "Device \[" | wc -l >%t.verbose.out
// RUN: %{run-unfiltered-devices} sycl-ls | wc -l >%t.concise.out
// RUN: %{run-aux} diff %t.verbose.out %t.concise.out

//==---- sycl-ls.cpp - SYCL test for consistency of sycl-ls output ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
