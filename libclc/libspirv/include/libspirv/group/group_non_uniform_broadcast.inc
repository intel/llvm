//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __GROUP_NONUNIFORM_BROADCAST(IdType)                                   \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE                         \
  __spirv_GroupNonUniformBroadcast(int Execution, __CLC_GENTYPE Value,         \
                                   IdType Id);

__GROUP_NONUNIFORM_BROADCAST(uint)
__GROUP_NONUNIFORM_BROADCAST(ulong)
#undef __GROUP_NONUNIFORM_BROADCAST
