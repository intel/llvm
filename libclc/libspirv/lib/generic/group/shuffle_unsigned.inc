//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_GEN_U)

#define __GROUP_SHUFFLE_NONUNIFORM(DIR)                                        \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE                         \
      __spirv_GroupNonUniformShuffle##DIR(int Execution, __CLC_GENTYPE Value,  \
                                          uint IdMaskDelta) {                  \
    return __CLC_AS_GENTYPE(__spirv_GroupNonUniformShuffle##DIR(               \
        Execution, __CLC_AS_S_GENTYPE(Value), IdMaskDelta));                   \
  }

__GROUP_SHUFFLE_NONUNIFORM()
__GROUP_SHUFFLE_NONUNIFORM(Xor)
__GROUP_SHUFFLE_NONUNIFORM(Up)
__GROUP_SHUFFLE_NONUNIFORM(Down)
#undef __GROUP_SHUFFLE_NONUNIFORM

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleINTEL(__CLC_GENTYPE Data, uint InvocationId) {
  return __CLC_AS_GENTYPE(
      __spirv_SubgroupShuffleINTEL(__CLC_AS_S_GENTYPE(Data), InvocationId));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleDownINTEL(__CLC_GENTYPE Current, __CLC_GENTYPE Next,
                                 uint Delta) {
  return __CLC_AS_GENTYPE(__spirv_SubgroupShuffleDownINTEL(
      __CLC_AS_S_GENTYPE(Current), __CLC_AS_S_GENTYPE(Next), Delta));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleUpINTEL(__CLC_GENTYPE Previous, __CLC_GENTYPE Current,
                               uint Delta) {
  return __CLC_AS_GENTYPE(__spirv_SubgroupShuffleUpINTEL(
      __CLC_AS_S_GENTYPE(Previous), __CLC_AS_S_GENTYPE(Current), Delta));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleXorINTEL(__CLC_GENTYPE Data, uint Value) {
  return __CLC_AS_GENTYPE(
      __spirv_SubgroupShuffleXorINTEL(__CLC_AS_S_GENTYPE(Data), Value));
}

#endif // defined(__CLC_GEN_U)
