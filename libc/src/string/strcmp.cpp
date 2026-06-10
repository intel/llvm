//===-- Implementation of strcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcmp.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_strcmp.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef __SPIRV__
#define __generic __attribute__((opencl_generic))
extern "C"
int strcmp( __generic const void* left,  __generic const void* right) {
  auto comp = [](char l, char r) -> int {
    return static_cast<unsigned char>(l) - static_cast<unsigned char>(r);
  };
  __generic const unsigned char * left_c = reinterpret_cast<__generic const unsigned char *>(left);
  __generic const unsigned char * right_c = reinterpret_cast<__generic const unsigned char *>(right);
  for (; *left_c && !comp(*left_c, *right_c); ++left_c, ++right_c)
    ;
  return comp(*left_c, *right_c);
}
#else
LLVM_LIBC_FUNCTION(int, strcmp, (const char *left, const char *right)) {
  auto comp = [](char l, char r) -> int {
    return static_cast<unsigned char>(l) - static_cast<unsigned char>(r);
  };
  return inline_strcmp(left, right, comp);
}
#endif
} // namespace LIBC_NAMESPACE_DECL
