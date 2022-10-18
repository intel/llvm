//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE int *ep) {
  int t_ep;
  __CLC_GENTYPE res = __clc_spirv_ocl_frexp(x, &t_ep);
  *ep = t_ep;
  return res;
}

#define __CLC_GENTYPE_VEC __CLC_XCONCAT(__CLC_GENTYPE, 2)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE_VEC
__spirv_ocl_frexp(__CLC_GENTYPE_VEC x, __CLC_ADDRESS_SPACE int2 *ep) {
  int ep_x;
  int ep_y;
  __CLC_GENTYPE_VEC res = (__CLC_GENTYPE_VEC)(__spirv_ocl_frexp(x.x, &ep_x),
                                              __spirv_ocl_frexp(x.y, &ep_y));
  *ep = (int2)(ep_x, ep_y);
  return res;
}

#undef __CLC_GENTYPE_VEC
#define __CLC_GENTYPE_VEC __CLC_XCONCAT(__CLC_GENTYPE, 3)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE_VEC
__spirv_ocl_frexp(__CLC_GENTYPE_VEC x, __CLC_ADDRESS_SPACE int3 *ep) {
  int ep_x;
  int ep_y;
  int ep_z;
  __CLC_GENTYPE_VEC res = (__CLC_GENTYPE_VEC)(__spirv_ocl_frexp(x.x, &ep_x),
                                              __spirv_ocl_frexp(x.y, &ep_y),
                                              __spirv_ocl_frexp(x.z, &ep_z));
  *ep = (int3)(ep_x, ep_y, ep_z);
  return res;
}

#define __CLC_VECTORIZE_FREXP_IMPL(GENTYPE, VEC_LEN, HALF_VEC_LEN)             \
  _CLC_OVERLOAD _CLC_DEF GENTYPE __spirv_ocl_frexp(                            \
      GENTYPE x, __CLC_ADDRESS_SPACE int##VEC_LEN *ep) {                       \
    int##HALF_VEC_LEN ep_lo;                                                   \
    int##HALF_VEC_LEN ep_hi;                                                   \
    GENTYPE res = (GENTYPE)(__spirv_ocl_frexp(x.lo, &ep_lo),                   \
                            __spirv_ocl_frexp(x.hi, &ep_hi));                  \
    *ep = (int##VEC_LEN)(ep_lo, ep_hi);                                        \
    return res;                                                                \
  }

#define __CLC_VECTORIZE_FREXP(VEC_LEN, HALF_VEC_LEN)                           \
  __CLC_VECTORIZE_FREXP_IMPL(__CLC_XCONCAT(__CLC_GENTYPE, VEC_LEN), VEC_LEN,   \
                             HALF_VEC_LEN)

__CLC_VECTORIZE_FREXP(4, 2)
__CLC_VECTORIZE_FREXP(8, 4)
__CLC_VECTORIZE_FREXP(16, 8)

#undef __CLC_GENTYPE_VEC
#undef __CLC_VECTORIZE_FREXP
