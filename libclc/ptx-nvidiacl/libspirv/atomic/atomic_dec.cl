//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_inc_dec_helpers.h>
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

__CLC_NVVM_ATOMIC_INCDEC(unsigned int, j, IDecrement, -1)
__CLC_NVVM_ATOMIC_INCDEC(unsigned long, m, IDecrement, -1)

#undef __CLC_NVVM_ATOMIC_INCDEC_IMPL
#undef __CLC_NVVM_ATOMIC_INCDEC
