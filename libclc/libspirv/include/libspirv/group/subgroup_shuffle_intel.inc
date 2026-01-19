//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleINTEL(__CLC_GENTYPE Data, uint InvocationId);

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleDownINTEL(__CLC_GENTYPE Current, __CLC_GENTYPE Next,
                                 uint Delta);

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleUpINTEL(__CLC_GENTYPE Previous, __CLC_GENTYPE Current,
                               uint Delta);

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT __CLC_GENTYPE
__spirv_SubgroupShuffleXorINTEL(__CLC_GENTYPE Data, uint Value);
