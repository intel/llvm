//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
#define __CLC_DECLARE_ATOMIC(ADDRSPACE, ADDRSPACE_MANGLED, TYPE, TYPE_MANGLED,                                                    \
                             NAME, NAME_LEN)                                                                                      \
  _CLC_DECL TYPE                                                                                                                  \
      _Z##NAME_LEN##NAME##PU3##ADDRSPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile ADDRSPACE TYPE *, enum Scope, enum MemorySemanticsMask,                                                        \
          TYPE);

#define __CLC_DECLARE_ATOMIC_ADDRSPACE(TYPE, TYPE_MANGLED, NAME, NAME_LEN) \
	__CLC_DECLARE_ATOMIC(global, AS1, TYPE, TYPE_MANGLED, NAME, NAME_LEN) \
	__CLC_DECLARE_ATOMIC(local, AS3, TYPE, TYPE_MANGLED, NAME, NAME_LEN)

__CLC_DECLARE_ATOMIC_ADDRSPACE(int, i, __SPIRV_FUNCTION_S, __SPIRV_FUNCTION_S_LEN)
__CLC_DECLARE_ATOMIC_ADDRSPACE(uint, j, __SPIRV_FUNCTION_U, __SPIRV_FUNCTION_U_LEN)

#ifdef __SPIRV_INT64_EXTENDED
#ifdef cl_khr_int64_extended_atomics
__CLC_DECLARE_ATOMIC_ADDRSPACE(long, l, __SPIRV_FUNCTION_S, __SPIRV_FUNCTION_S_LEN)
__CLC_DECLARE_ATOMIC_ADDRSPACE(ulong, m, __SPIRV_FUNCTION_U, __SPIRV_FUNCTION_U_LEN)
#endif
#endif

#ifdef __SPIRV_INT64_BASE
#ifdef cl_khr_int64_base_atomics
__CLC_DECLARE_ATOMIC_ADDRSPACE(long, l, __SPIRV_FUNCTION_S, __SPIRV_FUNCTION_S_LEN)
__CLC_DECLARE_ATOMIC_ADDRSPACE(ulong, m, __SPIRV_FUNCTION_U, __SPIRV_FUNCTION_U_LEN)
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
