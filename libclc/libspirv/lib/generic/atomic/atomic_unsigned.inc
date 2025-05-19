//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, nder the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_GEN_U) && defined(__CLC_SCALAR) &&                           \
    (__CLC_GENSIZE == 32 || __CLC_GENSIZE == 64)

#define __SPIRV_ATOMIC_3ARGS_IMPL(NAME, AS)                                    \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_Atomic##NAME(                   \
      AS __CLC_GENTYPE *Ptr, int Scope, int Semantics) {                       \
    return __CLC_AS_GENTYPE(                                                   \
        __spirv_Atomic##NAME((AS __CLC_S_GENTYPE *)Ptr, Scope, Semantics));    \
  }

#define __SPIRV_ATOMIC_4ARGS_IMPL(NAME, AS)                                    \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_Atomic##NAME(                   \
      AS __CLC_GENTYPE *Ptr, int Scope, int Semantics, __CLC_GENTYPE Value) {  \
    return __CLC_AS_GENTYPE(__spirv_Atomic##NAME((AS __CLC_S_GENTYPE *)Ptr,    \
                                                 Scope, Semantics,             \
                                                 __CLC_AS_S_GENTYPE(Value)));  \
  }

#define __SPIRV_ATOMIC_STORE(AS)                                               \
  _CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(                             \
      AS __CLC_GENTYPE *Ptr, int Scope, int Semantics, __CLC_GENTYPE Value) {  \
    __spirv_AtomicStore((AS __CLC_S_GENTYPE *)Ptr, Scope, Semantics,           \
                        __CLC_AS_S_GENTYPE(Value));                            \
  }

#define __SPIRV_ATOMIC_CMP_XCHG(AS)                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_AtomicCompareExchange(          \
      AS __CLC_GENTYPE *Ptr, int Scope, int Equal, int UnEqual,                \
      __CLC_GENTYPE Value, __CLC_GENTYPE Comparator) {                         \
    return __CLC_AS_GENTYPE(__spirv_AtomicCompareExchange(                     \
        (AS __CLC_S_GENTYPE *)Ptr, Scope, Equal, UnEqual,                      \
        __CLC_AS_S_GENTYPE(Value), __CLC_AS_S_GENTYPE(Comparator)));           \
  }

#if _CLC_GENERIC_AS_SUPPORTED

#define __SPIRV_ATOMIC_3ARGS(NAME)                                             \
  __SPIRV_ATOMIC_3ARGS_IMPL(NAME, local)                                       \
  __SPIRV_ATOMIC_3ARGS_IMPL(NAME, global)                                      \
  __SPIRV_ATOMIC_3ARGS_IMPL(NAME, )

#define __SPIRV_ATOMIC_4ARGS(NAME)                                             \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, local)                                       \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, global)                                      \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, )

#else // _CLC_GENERIC_AS_SUPPORTED

#define __SPIRV_ATOMIC_3ARGS(NAME)                                             \
  __SPIRV_ATOMIC_3ARGS_IMPL(NAME, local)                                       \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, global)

#define __SPIRV_ATOMIC_4ARGS(NAME)                                             \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, local)                                       \
  __SPIRV_ATOMIC_4ARGS_IMPL(NAME, global)

#endif // _CLC_GENERIC_AS_SUPPORTED

__SPIRV_ATOMIC_3ARGS(IDecrement)
__SPIRV_ATOMIC_3ARGS(IIncrement)
__SPIRV_ATOMIC_3ARGS(Load)

__SPIRV_ATOMIC_4ARGS(And)
__SPIRV_ATOMIC_4ARGS(Exchange)
__SPIRV_ATOMIC_4ARGS(IAdd)
__SPIRV_ATOMIC_4ARGS(ISub)
__SPIRV_ATOMIC_4ARGS(Or)
__SPIRV_ATOMIC_4ARGS(Xor)

__SPIRV_ATOMIC_STORE(local)
__SPIRV_ATOMIC_STORE(global)
#if _CLC_GENERIC_AS_SUPPORTED
__SPIRV_ATOMIC_STORE()
#endif // _CLC_GENERIC_AS_SUPPORTED

__SPIRV_ATOMIC_CMP_XCHG(local)
__SPIRV_ATOMIC_CMP_XCHG(global)
#if _CLC_GENERIC_AS_SUPPORTED
__SPIRV_ATOMIC_CMP_XCHG()
#endif // _CLC_GENERIC_AS_SUPPORTED

#endif // defined(__CLC_GEN_U) && defined(__CLC_SCALAR) && (__CLC_GENSIZE == 32
       // || __CLC_GENSIZE == 64)
