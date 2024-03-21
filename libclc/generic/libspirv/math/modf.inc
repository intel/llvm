//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define ZERO 0.0
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rlong, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define ZERO 0.0f
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rint, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define ZERO 0.0h
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rshort, __CLC_VECSIZE)
#endif

#ifdef __CLC_SCALAR
#undef __CLC_CONVERT_NATN
#define __CLC_CONVERT_NATN
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_modf(__CLC_GENTYPE x,
                                                      private __CLC_GENTYPE *iptr) {
  *iptr = __spirv_ocl_trunc(x);
  return __spirv_ocl_copysign(
      __CLC_CONVERT_NATN(__spirv_IsInf(x)) ? ZERO : x - *iptr, x);
}

#define MODF_DEF(addrspace)                                                    \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_modf(                       \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) {                        \
    private __CLC_GENTYPE private_iptr;                                        \
    __CLC_GENTYPE ret = __spirv_ocl_modf(x, &private_iptr);                    \
    *iptr = private_iptr;                                                      \
    return ret;                                                                \
  }

MODF_DEF(local);
MODF_DEF(global);

#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
MODF_DEF(generic);
#endif

#undef __CLC_CONVERT_NATN
#undef ZERO
