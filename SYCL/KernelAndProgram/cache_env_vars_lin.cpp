// No JITing for host devices and diffrent environment variables on linux and
// windows.
// REQUIRES: (level_zero || opencl) && linux

// RUN: rm -rf %t/cache_dir
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -DTARGET_IMAGE=INC100

// When no environment variables pointing cache directory are set the cache is
// disabled
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u HOME env -u XDG_CACHE_HOME SYCL_PI_TRACE=-1 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 env -u SYCL_CACHE_DIR env -u HOME env -u XDG_CACHE_HOME SYCL_PI_TRACE=-1 %t.out | FileCheck %s --check-prefixes=CHECK-BUILD

// When any of environment variables pointing to cache root is present cache is
// enabled
// RUN: rm -rf %t/cache_dir
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 XDG_CACHE_HOME=%t/cache_dir SYCL_PI_TRACE=-1 env -u SYCL_CACHE_DIR env -u HOME %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 XDG_CACHE_HOME=%t/cache_dir SYCL_PI_TRACE=-1 env -u SYCL_CACHE_DIR env -u HOME %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// RUN: rm -rf %t/cache_dir
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 env -u XDG_CACHE_HOME env -u HOME %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=%t/cache_dir SYCL_PI_TRACE=-1 env -u XDG_CACHE_HOME env -u HOME %t.out %t.out | FileCheck %s --check-prefixes=CHECK-CACHE
// RUN: rm -rf %t/cache_dir
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 HOME=%t/cache_dir SYCL_PI_TRACE=-1 env -u XDG_CACHE_HOME env -u SYCL_CACHE_DIR %t.out | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: %BE_RUN_PLACEHOLDER SYCL_CACHE_PERSISTENT=1 HOME=%t/cache_dir SYCL_PI_TRACE=-1 env -u XDG_CACHE_HOME env -u SYCL_CACHE_DIR %t.out | FileCheck %s --check-prefixes=CHECK-CACHE

// CHECK-BUILD-NOT: piProgramCreateWithBinary(
// CHECK-BUILD: piProgramCreate(
// CHECK-BUILD: piProgramBuild(

// CHECK-CACHE-NOT: piProgramCreate(
// CHECK-CACHE: piProgramCreateWithBinary(
// CHECK-CACHE: piProgramBuild(

#include "cache_env_vars.hpp"
