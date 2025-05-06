//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_GEN_U)

#define __GROUP_BROADCAST(LocalIdType)                                         \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE __spirv_GroupBroadcast( \
      int Execution, __CLC_GENTYPE Value, LocalIdType LocalId) {               \
    return __CLC_AS_GENTYPE(__spirv_GroupBroadcast(                            \
        Execution, __CLC_AS_S_GENTYPE(Value), LocalId));                       \
  }

__GROUP_BROADCAST(ulong)
__GROUP_BROADCAST(ulong2)
__GROUP_BROADCAST(ulong3)
__GROUP_BROADCAST(uint)
__GROUP_BROADCAST(uint2)
__GROUP_BROADCAST(uint3)
#undef __GROUP_BROADCAST

#define __GROUP_OPERATION(Name)                                                \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE __spirv_Group##Name(    \
      int Execution, int Operation, __CLC_GENTYPE X) {                         \
    return __CLC_AS_GENTYPE(                                                   \
        __spirv_Group##Name(Execution, Operation, __CLC_AS_S_GENTYPE(X)));     \
  }

__GROUP_OPERATION(IAdd)
__GROUP_OPERATION(IMulKHR)
__GROUP_OPERATION(BitwiseAndKHR)
__GROUP_OPERATION(BitwiseOrKHR)
__GROUP_OPERATION(BitwiseXorKHR)
#undef __GROUP_OPERATION

#define __GROUP_NONUNIFORM_BROADCAST(IdType)                                   \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE                         \
  __spirv_GroupNonUniformBroadcast(int Execution, __CLC_GENTYPE Value,         \
                                   IdType Id) {                                \
    return __CLC_AS_GENTYPE(__spirv_GroupNonUniformBroadcast(                  \
        Execution, __CLC_AS_S_GENTYPE(Value), Id));                            \
  }

__GROUP_NONUNIFORM_BROADCAST(uint)
__GROUP_NONUNIFORM_BROADCAST(ulong)

#define __GROUP_OPERATION_NONUNIFORM(Name)                                     \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE                         \
      __spirv_GroupNonUniform##Name(int Execution, int Operation,              \
                                    __CLC_GENTYPE Value, uint ClusterSize) {   \
    return __CLC_AS_GENTYPE(__spirv_GroupNonUniform##Name(                     \
        Execution, Operation, __CLC_AS_S_GENTYPE(Value), ClusterSize));        \
  }

__GROUP_OPERATION_NONUNIFORM(IAdd)
__GROUP_OPERATION_NONUNIFORM(IMul)
__GROUP_OPERATION_NONUNIFORM(BitwiseAnd)
__GROUP_OPERATION_NONUNIFORM(BitwiseOr)
__GROUP_OPERATION_NONUNIFORM(BitwiseXor)
#undef __GROUP_OPERATION_NONUNIFORM

#endif // defined(__CLC_GEN_U)
