//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF int __clc__atomic_load_global_4_unordered(global const int *);
_CLC_DEF int __clc__atomic_load_global_4_acquire(global const int *);
_CLC_DEF int __clc__atomic_load_global_4_seq_cst(global const int *);
_CLC_DEF int __clc__atomic_load_local_4_unordered(local const int *);
_CLC_DEF int __clc__atomic_load_local_4_acquire(local const int *);
_CLC_DEF int __clc__atomic_load_local_4_seq_cst(local const int *);

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicLoad(volatile global const int *p,
                                              unsigned int scope,
                                              unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_4_seq_cst(p);
  }
  return __clc__atomic_load_global_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicLoad(volatile local const int *p,
                                              unsigned int scope,
                                              unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_4_seq_cst(p);
  }
  return __clc__atomic_load_local_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicLoad(volatile global const unsigned int *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_4_seq_cst(p);
  }
  return __clc__atomic_load_global_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicLoad(volatile local const unsigned int *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_4_seq_cst(p);
  }
  return __clc__atomic_load_local_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicLoad(global const int *p,
                                              unsigned int scope,
                                              unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_4_seq_cst(p);
  }
  return __clc__atomic_load_global_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicLoad(local const int *p,
                                              unsigned int scope,
                                              unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_4_seq_cst(p);
  }
  return __clc__atomic_load_local_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicLoad(global const unsigned int *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_4_seq_cst(p);
  }
  return __clc__atomic_load_global_4_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicLoad(local const unsigned int *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_4_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_4_seq_cst(p);
  }
  return __clc__atomic_load_local_4_unordered(p);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_DEF long __clc__atomic_load_global_8_unordered(global const long *);
_CLC_DEF long __clc__atomic_load_global_8_acquire(global const long *);
_CLC_DEF long __clc__atomic_load_global_8_seq_cst(global const long *);
_CLC_DEF long __clc__atomic_load_local_8_unordered(local const long *);
_CLC_DEF long __clc__atomic_load_local_8_acquire(local const long *);
_CLC_DEF long __clc__atomic_load_local_8_seq_cst(local const long *);

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicLoad(volatile global const long *p,
                                               unsigned int scope,
                                               unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_8_seq_cst(p);
  }
  return __clc__atomic_load_global_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicLoad(volatile local const long *p,
                                               unsigned int scope,
                                               unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_8_seq_cst(p);
  }
  return __clc__atomic_load_local_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicLoad(volatile global const unsigned long *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_8_seq_cst(p);
  }
  return __clc__atomic_load_global_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicLoad(volatile local const unsigned long *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_8_seq_cst(p);
  }
  return __clc__atomic_load_local_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicLoad(global const long *p,
                                               unsigned int scope,
                                               unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_8_seq_cst(p);
  }
  return __clc__atomic_load_global_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicLoad(local const long *p,
                                               unsigned int scope,
                                               unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_8_seq_cst(p);
  }
  return __clc__atomic_load_local_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicLoad(global const unsigned long *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_global_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_global_8_seq_cst(p);
  }
  return __clc__atomic_load_global_8_unordered(p);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicLoad(local const unsigned long *p, unsigned int scope,
                   unsigned int semantics) {
  if (semantics == Acquire) {
    return __clc__atomic_load_local_8_acquire(p);
  }
  if (semantics == SequentiallyConsistent) {
    return __clc__atomic_load_local_8_seq_cst(p);
  }
  return __clc__atomic_load_local_8_unordered(p);
}
#endif // cl_khr_int64_extended_atomics
