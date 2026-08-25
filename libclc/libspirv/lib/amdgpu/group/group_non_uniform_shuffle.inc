//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 1

// Each operation maps directly onto the corresponding, already validated,
// __spirv_SubgroupShuffle*INTEL primitive (see misc/sub_group_shuffle.cl),
// which lowers to the hardware ds_bpermute wavefront shuffle. The Shuffle Up
// and Down primitives take a pair of (previous/current) or (current/next)
// operands; for the single-operand SPIR-V form we pass the same value for both,
// which is correct within a single sub-group and preserves the SPIR-V contract
// that out-of-range indices produce an undefined result.
// The execution operand is unused: the delegated __spirv_SubgroupShuffle*INTEL
// primitives operate at sub-group (wavefront) granularity, and the SPIR-V spec
// requires Execution to be Subgroup for these instructions.
_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_GroupNonUniformShuffle(int execution, __CLC_GENTYPE value, uint id) {
  (void)execution;
  return __spirv_SubgroupShuffleINTEL(value, id);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_GroupNonUniformShuffleXor(int execution, __CLC_GENTYPE value,
                                  uint mask) {
  (void)execution;
  return __spirv_SubgroupShuffleXorINTEL(value, mask);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_GroupNonUniformShuffleUp(int execution, __CLC_GENTYPE value,
                                 uint delta) {
  (void)execution;
  return __spirv_SubgroupShuffleUpINTEL(value, value, delta);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_GroupNonUniformShuffleDown(int execution, __CLC_GENTYPE value,
                                   uint delta) {
  (void)execution;
  return __spirv_SubgroupShuffleDownINTEL(value, value, delta);
}

#else

// SPIR-V permits a vector Value operand and clang declares these builtins for
// every vector width, so the vector forms are shuffled element-wise on top of
// the scalar ones above.
#define __CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE(FUNCTION, ARG_NAME)          \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE FUNCTION(               \
      int execution, __CLC_GENTYPE value, uint ARG_NAME) {                     \
    union {                                                                    \
      __CLC_GENTYPE vec;                                                       \
      __CLC_SCALAR_GENTYPE arr[__CLC_VECSIZE_OR_1];                            \
    } u_value, u_result;                                                       \
    u_value.vec = value;                                                       \
    for (int i = 0; i < __CLC_VECSIZE_OR_1; ++i)                               \
      u_result.arr[i] = FUNCTION(execution, u_value.arr[i], ARG_NAME);         \
    return u_result.vec;                                                       \
  }

__CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE(__spirv_GroupNonUniformShuffle, id)
__CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE(__spirv_GroupNonUniformShuffleXor,
                                          mask)
__CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE(__spirv_GroupNonUniformShuffleUp,
                                          delta)
__CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE(__spirv_GroupNonUniformShuffleDown,
                                          delta)

#undef __CLC_GROUP_NON_UNIFORM_SHUFFLE_SCALARIZE

#endif
