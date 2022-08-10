//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "membermask.h"

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

_CLC_DEF _CLC_CONVERGENT __clc_vec4_uint32_t
_Z29__spirv_GroupNonUniformBallotjb(unsigned flag, bool predicate) {
  // only support subgroup for now
  if (flag != Subgroup) {
    __builtin_trap();
    __builtin_unreachable();
  }

  // prepare result, we only support the ballot operation on 32 threads maximum
  // so we only need the first element to represent the final mask
  __clc_vec4_uint32_t res;
  res[1] = 0;
  res[2] = 0;
  res[3] = 0;

  // compute thread mask
  unsigned threads = __clc__membermask();

  // run the ballot operation
  res[0] = __nvvm_vote_ballot_sync(threads, predicate);

  return res;
}
