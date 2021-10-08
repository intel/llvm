// No JITing for host devices.
// Specialization constant values are not supported on CUDA
// REQUIRES: opencl || level_zero
// RUN: rm -rf %t/cache_dir
// FIXME Temporary disable fallback assert here until fixed
// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -DSYCL_DISABLE_FALLBACK_ASSERT=1 -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER --check-prefixes=CHECK-CACHE
//
// The test checks that caching works properly for SYCL application containing
// specialization constant values.
#include "spec_consts.hpp"

// CHECK-BUILD: piProgramBuild
// CHECK-BUILD-NOT: piProgramCreateWithBinary

// CHECK-CACHE-NOT: piProgramBuild
// CHECK-CACHE: piProgramCreateWithBinary
