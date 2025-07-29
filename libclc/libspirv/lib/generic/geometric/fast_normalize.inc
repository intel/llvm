//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SCALAR

// Only handle vector implementations
_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN __spirv_ocl_fast_normalize(__CLC_FLOATN p) {
  __CLC_FLOAT l2 = __spirv_Dot(p, p);
  return l2 == 0.0f ? p : p * __spirv_ocl_half_rsqrt(l2);
}

#endif
