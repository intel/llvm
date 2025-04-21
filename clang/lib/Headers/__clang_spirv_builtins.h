/*===---- spirv_builtin_vars.h - SPIR-V built-in ---------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __SPIRV_BUILTIN_VARS_H
#define __SPIRV_BUILTIN_VARS_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsycl-strict"

#pragma push_macro("SPIRV_NOEXCEPT")
#if __cplusplus >= 201103L
#define SPIRV_NOEXCEPT noexcept
#else
#define SPIRV_NOEXCEPT
#endif

#define __SPIRV_overloadable __attribute__((overloadable))
#define __SPIRV_convergent __attribute__((convergent))
#define __SPIRV_inline __attribute__((always_inline))

#define __SPIRV_purefn __attribute__((pure))
#define __SPIRV_cnfn __attribute__((const))

#define __global __attribute__((opencl_global))
#define __local __attribute__((opencl_local))
#define __private __attribute__((opencl_private))
#define __constant __attribute__((opencl_constant))
#ifdef __SYCL_DEVICE_ONLY__
#define __generic
#else
#define __generic __attribute__((opencl_generic))
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define __SPIRV_SYCL_DEVICE __attribute__((sycl_device))
#else
#define __SPIRV_SYCL_DEVICE
#endif

// Check if SPIR-V builtins are supported.
// As the translator doesn't use the LLVM intrinsics (which would be emitted if
// we use the SPIR-V builtins) we can't rely on the SPIRV32/SPIRV64 etc macros
// to establish if we can use the builtin alias. We disable builtin altogether
// if we do not intent to use the backend. So instead of use target macros, rely
// on a __has_builtin test.
#if (__has_builtin(__builtin_spirv_generic_cast_to_ptr_explicit))
#define _SPIRV_BUILTIN_ALIAS(builtin)                                          \
  __attribute__((clang_builtin_alias(builtin)))
#else
#define _SPIRV_BUILTIN_ALIAS(builtin)
#endif
#define _SPIRV_AVAILABILITY(platform, version)                                 \
  __attribute__((availability(platform, introduced = version)))
#define _SPIRV_AVAILABILITY_STAGE(platform, version, stage)                    \
  __attribute__((                                                              \
      availability(platform, introduced = version, environment = stage)))

// OpGenericCastToPtrExplicit

extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__global void *__spirv_GenericCastToPtrExplicit_ToGlobal(__generic void *,
                                                         int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__global const void *
__spirv_GenericCastToPtrExplicit_ToGlobal(__generic const void *,
                                          int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__global volatile void *
__spirv_GenericCastToPtrExplicit_ToGlobal(__generic volatile void *,
                                          int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__global const volatile void *
__spirv_GenericCastToPtrExplicit_ToGlobal(__generic const volatile void *,
                                          int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__local void *__spirv_GenericCastToPtrExplicit_ToLocal(__generic void *,
                                                       int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__local const void *
__spirv_GenericCastToPtrExplicit_ToLocal(__generic const void *,
                                         int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__local volatile void *
__spirv_GenericCastToPtrExplicit_ToLocal(__generic volatile void *,
                                         int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__local const volatile void *
__spirv_GenericCastToPtrExplicit_ToLocal(__generic const volatile void *,
                                         int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__private void *__spirv_GenericCastToPtrExplicit_ToPrivate(__generic void *,
                                                           int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__private const void *
__spirv_GenericCastToPtrExplicit_ToPrivate(__generic const void *,
                                           int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__private volatile void *
__spirv_GenericCastToPtrExplicit_ToPrivate(__generic volatile void *,
                                           int) SPIRV_NOEXCEPT;
extern __SPIRV_SYCL_DEVICE __SPIRV_overloadable
_SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit)
__private const volatile void *
__spirv_GenericCastToPtrExplicit_ToPrivate(__generic const volatile void *,
                                           int) SPIRV_NOEXCEPT;

// OpGenericCastToPtr

static __SPIRV_overloadable __SPIRV_inline __global void *
__spirv_GenericCastToPtr_ToGlobal(__generic void *p, int) SPIRV_NOEXCEPT {
  return (__global void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __global const void *
__spirv_GenericCastToPtr_ToGlobal(__generic const void *p, int) SPIRV_NOEXCEPT {
  return (__global const void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __global volatile void *
__spirv_GenericCastToPtr_ToGlobal(__generic volatile void *p,
                                  int) SPIRV_NOEXCEPT {
  return (__global volatile void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __global const volatile void *
__spirv_GenericCastToPtr_ToGlobal(__generic const volatile void *p,
                                  int) SPIRV_NOEXCEPT {
  return (__global const volatile void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __local void *
__spirv_GenericCastToPtr_ToLocal(__generic void *p, int) SPIRV_NOEXCEPT {
  return (__local void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __local const void *
__spirv_GenericCastToPtr_ToLocal(__generic const void *p, int) SPIRV_NOEXCEPT {
  return (__local const void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __local volatile void *
__spirv_GenericCastToPtr_ToLocal(__generic volatile void *p,
                                 int) SPIRV_NOEXCEPT {
  return (__local volatile void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __local const volatile void *
__spirv_GenericCastToPtr_ToLocal(__generic const volatile void *p,
                                 int) SPIRV_NOEXCEPT {
  return (__local const volatile void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __private void *
__spirv_GenericCastToPtr_ToPrivate(__generic void *p, int) SPIRV_NOEXCEPT {
  return (__private void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __private const void *
__spirv_GenericCastToPtr_ToPrivate(__generic const void *p,
                                   int) SPIRV_NOEXCEPT {
  return (__private const void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __private volatile void *
__spirv_GenericCastToPtr_ToPrivate(__generic volatile void *p,
                                   int) SPIRV_NOEXCEPT {
  return (__private volatile void *)p;
}
static __SPIRV_overloadable __SPIRV_inline __private const volatile void *
__spirv_GenericCastToPtr_ToPrivate(__generic const volatile void *p,
                                   int) SPIRV_NOEXCEPT {
  return (__private const volatile void *)p;
}

#undef __SPIRV_overloadable
#undef __SPIRV_convergent
#undef __SPIRV_inline

#undef __SPIRV_purefn
#undef __SPIRV_cnfn

#undef __global
#undef __local
#undef __constant
#undef __generic

#undef _SPIRV_BUILTIN_ALIAS
#undef _SPIRV_AVAILABILITY
#undef _SPIRV_AVAILABILITY_STAGE

#pragma pop_macro("SPIRV_NOEXCEPT")
#pragma clang diagnostic pop

#endif /* __SPIRV_BUILTIN_VARS_H */
