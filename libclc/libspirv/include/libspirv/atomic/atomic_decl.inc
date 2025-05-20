//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_DECLARE_ATOMIC(ADDRSPACE, TYPE, NAME)                            \
  _CLC_OVERLOAD _CLC_DECL TYPE NAME(ADDRSPACE TYPE *, int Scope,               \
                                    int MemorySemanticsMask, TYPE);

#define __CLC_DECLARE_ATOMIC_ADDRSPACE(TYPE, NAME)                             \
  __CLC_DECLARE_ATOMIC(global, TYPE, NAME)                                     \
  __CLC_DECLARE_ATOMIC(local, TYPE, NAME)                                      \
  __CLC_DECLARE_ATOMIC(, TYPE, NAME)

__CLC_DECLARE_ATOMIC_ADDRSPACE(int, __SPIRV_FUNCTION_S)
__CLC_DECLARE_ATOMIC_ADDRSPACE(uint, __SPIRV_FUNCTION_U)

#ifdef __SPIRV_INT64_EXTENDED
#ifdef cl_khr_int64_extended_atomics
__CLC_DECLARE_ATOMIC_ADDRSPACE(long, __SPIRV_FUNCTION_S)
__CLC_DECLARE_ATOMIC_ADDRSPACE(ulong, __SPIRV_FUNCTION_U)
#endif
#endif

#ifdef __SPIRV_INT64_BASE
#ifdef cl_khr_int64_base_atomics
__CLC_DECLARE_ATOMIC_ADDRSPACE(long, __SPIRV_FUNCTION_S)
__CLC_DECLARE_ATOMIC_ADDRSPACE(ulong, __SPIRV_FUNCTION_U)
#endif
#endif

#undef __CLC_DECLARE_ATOMIC_ADDRSPACE
#undef __CLC_DECLARE_ATOMIC

#undef __SPIRV_FUNCTION_S
#undef __SPIRV_FUNCTION_S_LEN
#undef __SPIRV_FUNCTION_U
#undef __SPIRV_FUNCTION_U_LEN
#undef __SPIRV_INT64_BASE
#undef __SPIRV_INT64_EXTENDED
