//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __GROUP_OPERATION(Name)                                                \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE __spirv_Group##Name(    \
      int Execution, int Operation, __CLC_GENTYPE X);

__GROUP_OPERATION(IAdd)
__GROUP_OPERATION(IMulKHR)
__GROUP_OPERATION(BitwiseAndKHR)
__GROUP_OPERATION(BitwiseOrKHR)
__GROUP_OPERATION(BitwiseXorKHR)
#undef __GROUP_OPERATION

#define __GROUP_OPERATION_NONUNIFORM(Name)                                     \
  _CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE                         \
      __spirv_GroupNonUniform##Name(int Execution, int Operation,              \
                                    __CLC_GENTYPE Value, uint ClusterSize);

__GROUP_OPERATION_NONUNIFORM(IAdd)
__GROUP_OPERATION_NONUNIFORM(IMul)
__GROUP_OPERATION_NONUNIFORM(BitwiseAnd)
__GROUP_OPERATION_NONUNIFORM(BitwiseOr)
__GROUP_OPERATION_NONUNIFORM(BitwiseXor)
#undef __GROUP_OPERATION_NONUNIFORM
