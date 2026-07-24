//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The __spirv_GroupNonUniform*Shuffle* builtins are emitted directly by the
// SYCL headers (see sycl/detail/spirv.hpp) for sub-group shuffle, permute and
// scan operations, and are relied upon by oneDPL work-group algorithms. Unlike
// SPIR-V targets, the amdgcn target has no runtime translation for these
// instructions, so they must be provided by the device library.
//
// The SYCL headers scalarize every shuffle before reaching the intrinsic
// (vectors and marrays are handled element-wise, and bitcast/generic shuffles
// are lowered onto integer scalars), so only scalar overloads are required
// here - mirroring the scalar-only __spirv_GroupBroadcast definitions in
// group/collectives.cl.
//
// Each operation maps directly onto the corresponding, already validated,
// __spirv_SubgroupShuffle*INTEL primitive (see misc/sub_group_shuffle.cl),
// which lowers to the hardware ds_bpermute wavefront shuffle. The Shuffle Up
// and Down primitives take a pair of (previous/current) or (current/next)
// operands; for the single-operand SPIR-V form we pass the same value for both,
// which is correct within a single sub-group and preserves the SPIR-V contract
// that out-of-range indices produce an undefined result.
// The scope operand is unused: the delegated __spirv_SubgroupShuffle*INTEL
// primitives operate at sub-group (wavefront) granularity, matching the
// Subgroup path of __spirv_GroupBroadcast in group/collectives.cl.
#define __CLC_GROUP_NON_UNIFORM_SHUFFLE(TYPE)                                  \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupNonUniformShuffle(  \
      int scope, TYPE value, uint id) {                                        \
    (void)scope;                                                               \
    return __spirv_SubgroupShuffleINTEL(value, id);                            \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE                                  \
      __spirv_GroupNonUniformShuffleXor(int scope, TYPE value, uint mask) {    \
    (void)scope;                                                               \
    return __spirv_SubgroupShuffleXorINTEL(value, mask);                       \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE                                  \
      __spirv_GroupNonUniformShuffleUp(int scope, TYPE value, uint delta) {    \
    (void)scope;                                                               \
    return __spirv_SubgroupShuffleUpINTEL(value, value, delta);                \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE                                  \
      __spirv_GroupNonUniformShuffleDown(int scope, TYPE value, uint delta) {  \
    (void)scope;                                                               \
    return __spirv_SubgroupShuffleDownINTEL(value, value, delta);              \
  }

__CLC_GROUP_NON_UNIFORM_SHUFFLE(char)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(uchar)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(short)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(ushort)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(int)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(uint)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(long)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(ulong)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(half)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(float)
__CLC_GROUP_NON_UNIFORM_SHUFFLE(double)

#undef __CLC_GROUP_NON_UNIFORM_SHUFFLE
