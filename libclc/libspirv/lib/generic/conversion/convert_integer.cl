//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <libspirv/spirv.h>

#define __CLC_S_SCALAR_TYPE_SRC __CLC_SCALAR_TYPE_SRC
#define __CLC_U_SCALAR_TYPE_SRC __CLC_XCONCAT(u, __CLC_SCALAR_TYPE_SRC)

#define __CLC_S_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_S_SCALAR_TYPE_SRC, __CLC_VECSIZE)
#define __CLC_U_GENTYPE_SRC                                                    \
  __CLC_XCONCAT(__CLC_U_SCALAR_TYPE_SRC, __CLC_VECSIZE)

#define __CLC_FUNCTION_S __CLC_XCONCAT(__spirv_SConvert_R, __CLC_GENTYPE)
#define __CLC_FUNCTION_U __CLC_XCONCAT(__spirv_UConvert_R, __CLC_GENTYPE)
#define __CLC_FUNCTION_S_SAT __CLC_XCONCAT(__CLC_FUNCTION_S, _sat)
#define __CLC_FUNCTION_U_SAT __CLC_XCONCAT(__CLC_FUNCTION_U, _sat)
#define __CLC_FUNCTION_SToU                                                    \
  __CLC_XCONCAT(__spirv_SatConvertSToU_R, __CLC_GENTYPE)
#define __CLC_FUNCTION_UToS                                                    \
  __CLC_XCONCAT(__spirv_SatConvertUToS_R, __CLC_GENTYPE)
#define __CLC_IMPL_FUNCTION __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE)
#define __CLC_IMPL_FUNCTION_SAT __CLC_XCONCAT(__CLC_IMPL_FUNCTION, _sat)

#define __CLC_SCALAR_TYPE_SRC char
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC short
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC int
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC

#define __CLC_SCALAR_TYPE_SRC long
#define __CLC_BODY <convert_integer.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_SCALAR_TYPE_SRC
