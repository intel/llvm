//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __GROUP_BROADCAST(LocalIdType)                                         \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE __spirv_GroupBroadcast( \
      int Execution, __CLC_GENTYPE Value, LocalIdType LocalId);

__GROUP_BROADCAST(ulong)
__GROUP_BROADCAST(ulong2)
__GROUP_BROADCAST(ulong3)
__GROUP_BROADCAST(uint)
__GROUP_BROADCAST(uint2)
__GROUP_BROADCAST(uint3)
#undef __GROUP_BROADCAST
