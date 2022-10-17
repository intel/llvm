//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef __CLC_SCALAR
#define __CLC_VECSIZE
#endif

#if __CLC_FPSIZE == 64
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(ulong, __CLC_VECSIZE) code) {
  return __CLC_AS_GENTYPE(code | 0x7ff8000000000000ul);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(long, __CLC_VECSIZE) code) {
  return __spirv_ocl_nan(__CLC_AS_UNSIGNED(ulong)(code));
}
#elif __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(uint, __CLC_VECSIZE) code) {
  return __CLC_AS_GENTYPE(code | 0x7fc00000);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(int, __CLC_VECSIZE) code) {
  return __spirv_ocl_nan(__CLC_AS_UNSIGNED(uint)(code));
}
#elif __CLC_FPSIZE == 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(ushort, __CLC_VECSIZE) code) {
  const ushort mask = 0x7e00;
  const __CLC_XCONCAT(ushort, __CLC_VECSIZE) res = code | mask;
  return __CLC_AS_GENTYPE(res);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_ocl_nan(__CLC_XCONCAT(short, __CLC_VECSIZE) code) {
  return __spirv_ocl_nan(__CLC_AS_UNSIGNED(ushort)(code));
}
#endif

#ifdef __CLC_SCALAR
#undef __CLC_VECSIZE
#endif
