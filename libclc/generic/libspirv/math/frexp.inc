//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_AS_GENTYPE __CLC_XCONCAT(as_, __CLC_GENTYPE)
#define __CLC_AS_INTN __CLC_XCONCAT(as_, __CLC_INTN)

#if __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_INTN i = __CLC_AS_INTN(x);
  __CLC_INTN ai = i & 0x7fffffff;
  __CLC_INTN d = ai > 0 & ai < 0x00800000;
  /* scale subnormal by 2^26 without multiplying */
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0d800000) - 0x1.0p-100f;
  ai = __spirv_ocl_select(ai, __CLC_AS_INTN(s), d);
  __CLC_INTN e =
      (ai >> 23) - 126 - __spirv_ocl_select((__CLC_INTN)0, (__CLC_INTN)26, d);
  __CLC_INTN t = ai == (__CLC_INTN)0 | e == (__CLC_INTN)129;
  i = (i & (__CLC_INTN)0x80000000) | (__CLC_INTN)0x3f000000 | (ai & 0x007fffff);
  *ep = __spirv_ocl_select(e, (__CLC_INTN)0, t);
  return __spirv_ocl_select(__CLC_AS_GENTYPE(i), x, t);
}
#endif

#if __CLC_FPSIZE == 64
#ifdef __CLC_SCALAR
#define __CLC_AS_LONGN as_long
#define __CLC_LONGN long
#define __CLC_CONVERT_INTN __spirv_SConvert_Rint
#else
#define __CLC_AS_LONGN __CLC_XCONCAT(as_long, __CLC_VECSIZE)
#define __CLC_LONGN __CLC_XCONCAT(long, __CLC_VECSIZE)
#define __CLC_CONVERT_INTN __CLC_XCONCAT(__spirv_SConvert_Rint, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_LONGN i = __CLC_AS_LONGN(x);
  __CLC_LONGN ai = i & 0x7fffffffffffffffL;
  __CLC_LONGN d = ai > 0 & ai < 0x0010000000000000L;
  // scale subnormal by 2^54 without multiplying
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0370000000000000L) - 0x1.0p-968;
  ai = __spirv_ocl_select(ai, __CLC_AS_LONGN(s), d);
  __CLC_LONGN e = (ai >> 52) - (__CLC_LONGN)1022 -
                  __spirv_ocl_select((__CLC_LONGN)0, (__CLC_LONGN)54, d);
  __CLC_LONGN t = ai == 0 | e == 1025;
  i = (i & (__CLC_LONGN)0x8000000000000000L) |
      (__CLC_LONGN)0x3fe0000000000000L |
      (ai & (__CLC_LONGN)0x000fffffffffffffL);
  *ep = __CLC_CONVERT_INTN(__spirv_ocl_select(e, 0L, t));
  return __spirv_ocl_select(__CLC_AS_GENTYPE(i), x, t);
}

#undef __CLC_AS_LONGN
#undef __CLC_LONGN
#undef __CLC_CONVERT_INTN
#endif

#undef __CLC_AS_GENTYPE
#undef __CLC_AS_INTN
