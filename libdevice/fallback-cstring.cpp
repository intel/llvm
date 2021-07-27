//==-- fallback-cstring.cpp - fallback implementation of C string functions--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wrapper.h"

#ifdef __SPIR__
void *__devicelib_memcpy(void *dest, const void *src, size_t n) {
  return __builtin_memcpy(dest, src, n);
}
#endif // __SPIR__
