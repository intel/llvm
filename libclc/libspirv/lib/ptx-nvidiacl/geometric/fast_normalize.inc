//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

#ifndef __CLC_SCALAR

// Only handle vector implementations
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_fast_normalize(__CLC_GENTYPE p) {
  __CLC_SCALAR_GENTYPE l2 = __spirv_Dot(p, p);
  return l2 == 0.0f ? p : p * __spirv_ocl_half_rsqrt(l2);
}

#endif
#endif
