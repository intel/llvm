//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memory_utils/inline_memcmp.h"

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {
#ifdef __SPIRV__

#define __generic __attribute__((opencl_generic))
extern "C"
int memcmp( __generic const void* lhs,  __generic const void* rhs, size_t count) {
  return __builtin_memcmp(lhs, rhs, count);
}
#else
LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  if (count) {
    LIBC_CRASH_ON_NULLPTR(lhs);
    LIBC_CRASH_ON_NULLPTR(rhs);
  }
  return inline_memcmp(lhs, rhs, count);
}
#endif

} // namespace LIBC_NAMESPACE_DECL
