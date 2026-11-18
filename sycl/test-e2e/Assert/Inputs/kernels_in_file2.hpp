//===-------------------------kernels_in_file2.hpp-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares device functions and kernel submission helpers shared by the
/// multi-translation-unit assertion tests.
/// An optional latch coordinates host threads to increase the likelihood of
/// simultaneous kernel submissions in multithreaded tests.
///
//===----------------------------------------------------------------------===//
#include <latch>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

SYCL_EXTERNAL int calculus(int X);

void enqueueKernel_1_fromFile2(sycl::queue *Q, std::latch *Sync = nullptr);

void enqueueKernel_2_fromFile2(sycl::queue *Q, std::latch *Sync = nullptr);
