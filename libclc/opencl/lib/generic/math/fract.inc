//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define MIN_CONSTANT 0x1.fffffffffffffp-1
#define ZERO 0.0
#elif __CLC_FPSIZE == 32
#define MIN_CONSTANT 0x1.fffffep-1f
#define ZERO 0.0f
#elif __CLC_FPSIZE == 16
#define MIN_CONSTANT 0x1.ffcp-1h
#define ZERO 0.0h
#endif

#define FRACT_DEF(addrspace) \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fract(__CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) { \
    return __spirv_ocl_fract(x, iptr); \
 }

FRACT_DEF(private);
FRACT_DEF(local);
FRACT_DEF(global);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
FRACT_DEF(generic);
#endif

#undef MIN_CONSTANT
#undef ZERO
