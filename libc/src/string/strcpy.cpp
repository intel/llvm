//===-- Implementation of strcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef __SPIRV__
#define __generic __attribute__((opencl_generic))
extern "C"
__generic char * strcpy(__generic char * __restrict dest, __generic char * __restrict src) {
  size_t size = internal::string_length(src) + 1;
  __builtin_memcpy(dest, src, size);
  return dest;
}
#else
LLVM_LIBC_FUNCTION(char *, strcpy,
                   (char *__restrict dest, const char *__restrict src)) {
  LIBC_CRASH_ON_NULLPTR(dest);
  size_t size = internal::string_length(src) + 1;
  inline_memcpy(dest, src, size);
  return dest;
}
#endif

} // namespace LIBC_NAMESPACE_DECL
