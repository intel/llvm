//==--- fallback-cassert.cpp - device agnostic implementation of C assert --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/spir_global_var.hpp"
#include "wrapper.h"

#if defined(__SPIR__) || defined(__SPIRV__)

extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);

static const __SYCL_CONSTANT__ char __assert_fmt[] =
   "%s:%d: %s: global id: [%lld, %lld, %lld], "
      "local id: [%lld, %lld ,%lld] "
      "Assertion `%s` failed.\n";
DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
 __spirv_ocl_printf(__assert_fmt, file, (int32_t)line,
                     // WORKAROUND: IGC does not handle this well
                     // (func) ? func : "<unknown function>",
                     func, gid0, gid1, gid2, lid0, lid1, lid2, expr);
}
#endif // __SPIR__ || __SPIRV__

#if defined(__NVPTX__) || defined(__AMDGCN__)

DEVICE_EXTERN_C void __assertfail(const char *__message, const char *__file,
                                  unsigned __line, const char *__function,
                                  size_t charSize);

DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
  __assertfail(expr, file, line, func, 1);
}

DEVICE_EXTERN_C void _wassert(const char *_Message, const char *_File,
                              unsigned _Line) {
  __assertfail(_Message, _File, _Line, 0, 1);
}

#endif // __NVPTX__ || __AMDGCN__
