// No JITing for host devices and diffrent environment variables on linux and
// windows.
// REQUIRES: (level_zero || opencl) && linux

// RUN: rm -rf %t/cache_dir
// RUN: %{build} -o %t.out -DTARGET_IMAGE=INC100

// When no environment variables pointing cache directory are set the cache is
// disabled
// RUN: env SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u HOME env -u XDG_CACHE_HOME SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u HOME env -u XDG_CACHE_HOME SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD

// When any of environment variables pointing to cache root is present cache is
// enabled
// RUN: rm -rf %t/cache_dir
// RUN: env SYCL_CACHE_PERSISTENT=1 XDG_CACHE_HOME=%t/cache_dir SYCL_UR_TRACE=2 env -u SYCL_CACHE_DIR env -u HOME %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 XDG_CACHE_HOME=%t/cache_dir SYCL_UR_TRACE=2 env -u SYCL_CACHE_DIR env -u HOME %{run} %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// RUN: rm -rf %t/cache_dir
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 env -u XDG_CACHE_HOME env -u HOME %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_UR_TRACE=2 env -u XDG_CACHE_HOME env -u HOME %{run} %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// RUN: rm -rf %t/cache_dir
// RUN: env SYCL_CACHE_PERSISTENT=1 HOME=%t/cache_dir SYCL_UR_TRACE=2 env -u XDG_CACHE_HOME env -u SYCL_CACHE_DIR %{run} %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: env SYCL_CACHE_PERSISTENT=1 HOME=%t/cache_dir SYCL_UR_TRACE=2 env -u XDG_CACHE_HOME env -u SYCL_CACHE_DIR %{run} %t.out | FileCheck %s --check-prefixes=CHECK-CACHE

// Some backends will call urProgramBuild and some will call urProgramBuildExp depending on urProgramBuildExp support.

// CHECK-BUILD-NOT: <--- urProgramCreateWithBinary(
// CHECK-BUILD: <--- urProgramCreateWithIL(
// CHECK-BUILD: <--- urProgramBuild{{(Exp)?}}(

// CHECK-CACHE-NOT: <--- urProgramCreateWithIL(
// CHECK-CACHE: <--- urProgramCreateWithBinary(
// CHECK-CACHE: <--- urProgramBuild{{(Exp)?}}(

#include "cache_env_vars.hpp"
