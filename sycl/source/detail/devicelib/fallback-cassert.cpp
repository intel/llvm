//==--- fallback-cassert.cpp - device agnostic implementation of C assert --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wrapper.h"

// __attribute((format(...))) enables compiler checks for a format string.
int __spirv_ocl_printf(const  __attribute__((opencl_constant)) char* fmt, ...) __attribute__((format(printf, 1, 2)));

static const __attribute__((opencl_constant)) char assert_fmt[] =
    "%s:%d: %s: local id: [%lu,%lu,%lu], global id: [%lu,%lu,%lu] "
    "Assertion `%s` failed.\n";

SYCL_EXTERNAL
extern "C" void __devicelib_assert_fail(const char *expr, const char *file,
                                        int32_t line, const char *func,
                                        size_t gid0, size_t gid1, size_t gid2,
                                        size_t lid0, size_t lid1, size_t lid2) {
  // intX_t types are used instead of `int' and `long' because the format string
  // is defined in terms of *device* types (OpenCL types): %d matches a 32 bit
  // integer, %lu matches a 64 bit unsigned integer. Host `int' and
  // `long' types may be different, so we cannot use them.
  __spirv_ocl_printf(
      assert_fmt,
      file, (int32_t)line,
      (func) ? func : "<unknown function>",
      (uint64_t)gid0, (uint64_t)gid1, (uint64_t)gid2,
      (uint64_t)lid0, (uint64_t)lid1, (uint64_t)lid2,
      expr);

  // FIXME: call SPIR-V unreachable instead
  // volatile int *die = (int *)0x0;
  // *die = 0xdead;
}
