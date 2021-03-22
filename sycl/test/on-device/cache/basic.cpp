// No JITing for host devices.
// REQUIRES: opencl || level_zero || cuda
// RUN: rm -rf %T/cache_dir
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_DIR=%T/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
//
//==----------- basic.cpp --------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that caching works properly.
#include "basic.hpp"

// CHECK-BUILD: piProgramBuild
// CHECK-BUILD: piProgramCreateWithBinary

// CHECK-CACHE-NOT: piProgramBuild
// CHECK-CACHE: piProgramCreateWithBinary
