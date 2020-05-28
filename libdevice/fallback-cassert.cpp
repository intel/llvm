//==--- fallback-cassert.cpp - device agnostic implementation of C assert --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wrapper.h"

#ifdef __SPIR__
static const __attribute__((opencl_constant)) char assert_fmt[] =
    "%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] "
    "Assertion `%s` failed.\n";

DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
  // intX_t types are used instead of `int' and `long' because the format string
  // is defined in terms of *device* types (OpenCL types): %d matches a 32 bit
  // integer, %lu matches a 64 bit unsigned integer. Host `int' and
  // `long' types may be different, so we cannot use them.
  __spirv_ocl_printf(assert_fmt, file, (int32_t)line,
                     // WORKAROUND: IGC does not handle this well
                     // (func) ? func : "<unknown function>",
                     func, gid0, gid1, gid2, lid0, lid1, lid2, expr);

  // FIXME: call SPIR-V unreachable instead
  // volatile int *die = (int *)0x0;
  // *die = 0xdead;
}
#endif // __SPIR__
