//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_ULONGN code) {
  return __CLC_AS_GENTYPE(code | 0x7ff8000000000000ul);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_LONGN code) {
  return __spirv_ocl_nan(__CLC_AS_ULONGN(code));
}
#elif __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_UINTN code) {
  return __CLC_AS_GENTYPE(code | 0x7fc00000);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_INTN code) {
  return __spirv_ocl_nan(__CLC_AS_UINTN(code));
}
#elif __CLC_FPSIZE == 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_USHORTN code) {
  const ushort mask = 0x7e00;
  const __CLC_USHORTN res = code | mask;
  return __CLC_AS_GENTYPE(res);
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_nan(__CLC_SHORTN code) {
  return __spirv_ocl_nan(__CLC_AS_USHORTN(code));
}
#endif
