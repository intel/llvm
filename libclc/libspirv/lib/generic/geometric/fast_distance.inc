//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_FLOAT __spirv_ocl_fast_distance(__CLC_FLOATN p0,
                                                             __CLC_FLOATN p1) {
  return __spirv_ocl_fast_length(p0 - p1);
}
