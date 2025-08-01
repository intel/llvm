//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_BuiltInSubgroupMaxSize() {
  return 32;
  // FIXME: warpsize is defined by NVVM IR but doesn't compile if used here
  // return __nvvm_read_ptx_sreg_warpsize();
}
