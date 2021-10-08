// No JITing for host devices.
// REQUIRES: opencl || level_zero
// RUN: rm -rf %t/cache_dir
// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
//
// The test checks that caching works properly.
#include "basic.hpp"

// CHECK-BUILD: piProgramBuild

// CHECK-CACHE-NOT: piProgramBuild
// CHECK-CACHE: piProgramCreateWithBinary
