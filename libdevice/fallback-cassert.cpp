//==--- fallback-cassert.cpp - device agnostic implementation of C assert --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic.hpp"
#include "wrapper.h"

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
