//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------

#if defined(__CLC_SCALAR) && (defined(__CLC_FPSIZE) || (__CLC_GENSIZE >= 32))

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      ADDRSPACE __CLC_GENTYPE *Ptr, int Scope, int Semantics) {                \
    __CLC_GENTYPE Res =                                                        \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr,                    \
                            __spirv_get_clang_memory_order(Semantics),         \
                            __spirv_get_clang_memory_scope(Scope));            \
    return Res;                                                                \
  }
#elif defined(__CLC_INC_DEC)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      ADDRSPACE __CLC_GENTYPE *Ptr, int Scope, int Semantics) {                \
    __CLC_GENTYPE Res =                                                        \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr,                    \
                            __spirv_get_clang_memory_order(Semantics),         \
                            __spirv_get_clang_memory_scope(Scope));            \
    return Res;                                                                \
  }
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF void __CLC_FUNCTION(ADDRSPACE __CLC_GENTYPE *Ptr,     \
                                             int Scope, int Semantics,         \
                                             __CLC_GENTYPE Value) {            \
    __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Value,                 \
                        __spirv_get_clang_memory_order(Semantics),             \
                        __spirv_get_clang_memory_scope(Scope));                \
  }
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      ADDRSPACE __CLC_GENTYPE *Ptr, int Scope, int Equal, int Unequal,         \
      __CLC_GENTYPE Value, __CLC_GENTYPE Comparator) {                         \
    __CLC_GENTYPE Res =                                                        \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Comparator, Value, \
                            __spirv_get_clang_memory_order(Equal),             \
                            __spirv_get_clang_memory_order(Unequal),           \
                            __spirv_get_clang_memory_scope(Scope));            \
    return Res;                                                                \
  }
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      ADDRSPACE __CLC_GENTYPE *Ptr, int Scope, int Semantics,                  \
      __CLC_GENTYPE Value) {                                                   \
    __CLC_GENTYPE Res =                                                        \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Value,             \
                            __spirv_get_clang_memory_order(Semantics),         \
                            __spirv_get_clang_memory_scope(Scope));            \
    return Res;                                                                \
  }
#endif

__CLC_DEFINE_ATOMIC(private)
__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // defined(__CLC_SCALAR) && (defined(__CLC_FPSIZE) || (__CLC_GENSIZE >=
       // 32))
