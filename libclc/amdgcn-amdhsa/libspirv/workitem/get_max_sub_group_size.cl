//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// The clang driver will define this variable depending on the architecture and
// compile flags by linking in ROCm bitcode defining it to true or false. If
// it's 1 the wavefront size used is 64, if it's 0 the wavefront size used is
// 32.
extern constant unsigned char __oclc_wavefrontsize64;

_CLC_DEF _CLC_OVERLOAD uint __spirv_SubgroupMaxSize() {
  if (__oclc_wavefrontsize64 == 1) {
    return 64;
  }
  return 32;
}
