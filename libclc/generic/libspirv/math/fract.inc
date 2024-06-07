//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define MIN_CONSTANT 0x1.fffffffffffffp-1
#define ZERO (__CLC_GENTYPE)0.0
#define __CLC_BOOLN __CLC_XCONCAT(long, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define MIN_CONSTANT 0x1.fffffep-1f
#define ZERO (__CLC_GENTYPE)0.0f
#define __CLC_BOOLN __CLC_XCONCAT(int, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define MIN_CONSTANT 0x1.ffcp-1h
#define ZERO (__CLC_GENTYPE)0.0h
#define __CLC_BOOLN __CLC_XCONCAT(short, __CLC_VECSIZE)
#endif

#ifdef __CLC_SCALAR
#define __CLC_CONVERT
#else
#define __CLC_CONVERT __CLC_XCONCAT(__spirv_SConvert_R, __CLC_BOOLN)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_fract(__CLC_GENTYPE x, private __CLC_GENTYPE *iptr) {
  *iptr = __spirv_ocl_floor(x);
  __CLC_GENTYPE r = __spirv_ocl_fmin(x - *iptr, MIN_CONSTANT);
  r = __CLC_CONVERT(__spirv_IsInf(x)) ? ZERO : r;
  r = __CLC_CONVERT(__spirv_IsNan(x)) ? x : r;
  return r;
}

#define FRACT_DEF(addrspace)                                                   \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_fract(                      \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) {                        \
    private __CLC_GENTYPE private_iptr;                                        \
    __CLC_GENTYPE ret = __spirv_ocl_fract(x, &private_iptr);                   \
    *iptr = private_iptr;                                                      \
    return ret;                                                                \
  }

FRACT_DEF(local);
FRACT_DEF(global);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
FRACT_DEF(generic);
#endif

#undef MIN_CONSTANT
#undef ZERO
#undef __CLC_CONVERT
#undef __CLC_BOOLN
