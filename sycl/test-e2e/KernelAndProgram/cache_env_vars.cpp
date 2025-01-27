// No JITing for host devices.
// REQUIRES: opencl || level_zero
// RUN: rm -rf %t/cache_dir
// RUN: %{build} -o %t.out -DTARGET_IMAGE=INC100
// Build program and add item to cache
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Ignore caching because image size is less than threshold
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE=1000000 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Ignore caching because image size is more than threshold
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE=1000 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// Use cache
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// Ignore cache because of environment variable
// RUN: env SYCL_CACHE_PERSISTENT=0 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
//
// The test checks environment variables which may disable caching.
// Also it can be used for benchmarking cache:
// Rough data collected on 32 core machine.
// Number of lines    1      10    100    1000   10000
// Image Size(kB)     2       2	    20	   165    1700
// Device code build time in seconds
// CPU OCL JIT       0.12    0.12  0.16     1.1     16
// CPU OCL Cache     0.01    0.01  0.01	   0.02   0.08

// Some backends will call urProgramBuild and some will call
// urProgramBuildExp depending on urProgramBuildExp support.

// CHECK-BUILD-NOT: <--- urProgramCreateWithBinary(
// CHECK-BUILD: <--- urProgramCreateWithIL(
// CHECK-BUILD: <--- urProgramBuild{{(Exp)?}}(

// CHECK-CACHE-NOT: <--- urProgramCreateWithIL(
// CHECK-CACHE: <--- urProgramCreateWithBinary(
// CHECK-CACHE: <--- urProgramBuild{{(Exp)?}}(

#include "cache_env_vars.hpp"
