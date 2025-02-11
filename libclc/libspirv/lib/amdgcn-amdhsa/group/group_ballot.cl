//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

// from llvm/include/llvm/IR/InstrTypes.h
#define ICMP_NE 33

_CLC_DEF _CLC_CONVERGENT __clc_vec4_uint32_t
_Z29__spirv_GroupNonUniformBallotjb(unsigned flag, bool predicate) {
  // only support subgroup for now
  if (flag != Subgroup) {
    __builtin_trap();
    __builtin_unreachable();
  }

  // prepare result, we only support the ballot operation on 64 threads maximum
  // so we only need the first two elements to represent the final mask
  __clc_vec4_uint32_t res;
  res[2] = 0;
  res[3] = 0;

  // run the ballot operation
  res.xy = __builtin_amdgcn_uicmp((int)predicate, 0, ICMP_NE);

  return res;
}
