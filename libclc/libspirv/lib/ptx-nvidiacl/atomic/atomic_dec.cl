//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_inc_dec_helpers.h>
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

__CLC_NVVM_ATOMIC_INCDEC(unsigned int, IDecrement, -1)
__CLC_NVVM_ATOMIC_INCDEC(unsigned long, IDecrement, -1)

#undef __CLC_NVVM_ATOMIC_INCDEC_IMPL
#undef __CLC_NVVM_ATOMIC_INCDEC
