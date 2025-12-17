//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_GEN_U

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__spirv_BitFieldUExtract(__CLC_GENTYPE base, uint offset, uint count) {
  return __clc_bitfield_extract_unsigned(base, offset, count);
}

#endif
