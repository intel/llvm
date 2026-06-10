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
  size_t offset = 0;
  __generic const unsigned char* p1 = reinterpret_cast<__generic const unsigned char*>(lhs);
  __generic const unsigned char* p2 = reinterpret_cast<__generic const unsigned char*>(rhs);
  for (; offset < count; ++offset) {
    const int32_t a = static_cast<int32_t>(p1[offset]);
    const int32_t b = static_cast<int32_t>(p2[offset]);
    const int32_t diff = a - b;
    if (diff)
      return diff;
  }
  return 0;
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
