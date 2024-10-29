// No JITing for host devices and diffrent environment variables on linux and
// windows.
// REQUIRES: (level_zero || opencl) && windows

// RUN: rm -rf %t/cache_dir
// RUN: %{build} -o %t.out -DTARGET_IMAGE=INC100

// When no environment variables pointing cache directory are set the cache is
// disabled
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u AppData SYCL_UR_TRACE=2 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u AppData SYCL_UR_TRACE=2 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD

// When any of environment variables pointing to cache root is present cache is
// enabled
// RUN: rm -rf %t/cache_dir
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 env -u AppData %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 env -u AppData %t.out %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// RUN: rm -rf %t/cache_dir
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 AppData=%t/cache_dir SYCL_UR_TRACE=2 env -u SYCL_CACHE_DIR %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %{run-unfiltered-devices} env SYCL_CACHE_PERSISTENT=1 AppData=%t/cache_dir SYCL_UR_TRACE=2 env -u SYCL_CACHE_DIR %t.out | FileCheck %s --check-prefixes=CHECK-CACHE

// CHECK-BUILD-NOT: <--- urProgramCreateWithBinary(
// CHECK-BUILD: <--- urProgramCreateWithIL(
// CHECK-BUILD: <--- urProgramBuild{{(Exp)?}}(

// CHECK-CACHE-NOT: <--- urProgramCreateWithIL(
// CHECK-CACHE: <--- urProgramCreateWithBinary(
// CHECK-CACHE: <--- urProgramBuild{{(Exp)?}}(

#include "cache_env_vars.hpp"
