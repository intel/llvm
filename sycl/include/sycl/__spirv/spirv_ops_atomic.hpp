//==------- spirv_ops_atomic.hpp --- SPIRV atomic operations --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_builtin_decls.hpp>

#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__

// Atomic SPIR-V builtins
// TODO: drop these forward-declarations.
// As of now, compiler does not forward-declare long long overloads for
// these and as such we can't drop anything from here. But ideally, we should
// rely on the compiler to generate those - that would allow to drop
// spirv_ops.hpp include from more files.
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicLoad(AS Type *P, int S,      \
                                                       int O) noexcept;
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern __DPCPP_SYCL_EXTERNAL void __spirv_AtomicStore(                       \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicExchange(                    \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicCompareExchange(             \
      AS Type *P, int S, int E, int U, Type V, Type C) noexcept;
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicIAdd(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicISub(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FADD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFAddEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicSMin(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicUMin(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFMinEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicSMax(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicUMax(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFMaxEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicAnd(AS Type *P, int S,       \
                                                      int O, Type V) noexcept;
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicOr(AS Type *P, int S, int O, \
                                                     Type V) noexcept;
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicXor(AS Type *P, int S,       \
                                                      int O, Type V) noexcept;

#define __SPIRV_ATOMIC_FLOAT(AS, Type)                                         \
  __SPIRV_ATOMIC_FADD(AS, Type)                                                \
  __SPIRV_ATOMIC_FMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_FMAX(AS, Type)                                                \
  __SPIRV_ATOMIC_LOAD(AS, Type)                                                \
  __SPIRV_ATOMIC_STORE(AS, Type)                                               \
  __SPIRV_ATOMIC_EXCHANGE(AS, Type)

#define __SPIRV_ATOMIC_BASE(AS, Type)                                          \
  __SPIRV_ATOMIC_FLOAT(AS, Type)                                               \
  __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                        \
  __SPIRV_ATOMIC_IADD(AS, Type)                                                \
  __SPIRV_ATOMIC_ISUB(AS, Type)                                                \
  __SPIRV_ATOMIC_AND(AS, Type)                                                 \
  __SPIRV_ATOMIC_OR(AS, Type)                                                  \
  __SPIRV_ATOMIC_XOR(AS, Type)

#define __SPIRV_ATOMIC_SIGNED(AS, Type)                                        \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_SMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_SMAX(AS, Type)

#define __SPIRV_ATOMIC_UNSIGNED(AS, Type)                                      \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_UMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_UMAX(AS, Type)

// Helper atomic operations which select correct signed/unsigned version
// of atomic min/max based on the type
#define __SPIRV_ATOMIC_MINMAX(AS, Op)                                          \
  template <typename T>                                                        \
  typename std::enable_if_t<                                                   \
      std::is_integral<T>::value && std::is_signed<T>::value, T>               \
  __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics, T Value) noexcept { \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if_t<                                                   \
      std::is_integral<T>::value && !std::is_signed<T>::value, T>              \
  __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics, T Value) noexcept { \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if_t<std::is_floating_point<T>::value, T>               \
  __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics, T Value) noexcept { \
    return __spirv_AtomicF##Op##EXT(Ptr, Memory, Semantics, Value);            \
  }

#define __SPIRV_ATOMICS(macro, Arg)                                            \
  macro(__attribute__((opencl_global)), Arg)                                   \
      macro(__attribute__((opencl_local)), Arg) macro(, Arg)

__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, _Float16)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, float)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, double)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Min)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Max)

#undef __SPIRV_ATOMICS
#undef __SPIRV_ATOMIC_AND
#undef __SPIRV_ATOMIC_BASE
#undef __SPIRV_ATOMIC_CMP_EXCHANGE
#undef __SPIRV_ATOMIC_EXCHANGE
#undef __SPIRV_ATOMIC_FADD
#undef __SPIRV_ATOMIC_FLOAT
#undef __SPIRV_ATOMIC_FMAX
#undef __SPIRV_ATOMIC_FMIN
#undef __SPIRV_ATOMIC_IADD
#undef __SPIRV_ATOMIC_ISUB
#undef __SPIRV_ATOMIC_LOAD
#undef __SPIRV_ATOMIC_MINMAX
#undef __SPIRV_ATOMIC_OR
#undef __SPIRV_ATOMIC_SIGNED
#undef __SPIRV_ATOMIC_SMAX
#undef __SPIRV_ATOMIC_SMIN
#undef __SPIRV_ATOMIC_STORE
#undef __SPIRV_ATOMIC_UMAX
#undef __SPIRV_ATOMIC_UMIN
#undef __SPIRV_ATOMIC_UNSIGNED
#undef __SPIRV_ATOMIC_XOR

#endif