//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_GEN_S

#ifdef __CLC_ATOMIC_MIN
#define __CLC_FUNCTION __spirv_AtomicSMin
#else
#define __CLC_FUNCTION __spirv_AtomicSMax
#endif

#else

#ifdef __CLC_ATOMIC_MIN
#define __CLC_FUNCTION __spirv_AtomicUMin
#else
#define __CLC_FUNCTION __spirv_AtomicUMax
#endif

#endif

#include <atomic_def.inc>

#undef __CLC_FUNCTION
