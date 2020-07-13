//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF int __clc__atomic_store_global_4_unordered(global const int *, int);
_CLC_DEF int __clc__atomic_store_global_4_release(global const int *, int);
_CLC_DEF int __clc__atomic_store_global_4_seq_cst(global const int *, int);
_CLC_DEF int __clc__atomic_store_local_4_unordered(local const int *, int);
_CLC_DEF int __clc__atomic_store_local_4_release(local const int *, int);
_CLC_DEF int __clc__atomic_store_local_4_seq_cst(local const int *, int);
_CLC_DEF unsigned int
__clc__atomic_ustore_global_4_unordered(global const unsigned int *,
                                        unsigned int);
_CLC_DEF unsigned int
__clc__atomic_ustore_global_4_release(global const unsigned int *,
                                      unsigned int);
_CLC_DEF unsigned int
__clc__atomic_ustore_global_4_seq_cst(global const unsigned int *,
                                      unsigned int);
_CLC_DEF unsigned int
__clc__atomic_ustore_local_4_unordered(local const unsigned int *,
                                       unsigned int);
_CLC_DEF unsigned int
__clc__atomic_ustore_local_4_release(local const unsigned int *, unsigned int);
_CLC_DEF unsigned int
__clc__atomic_ustore_local_4_seq_cst(local const unsigned int *, unsigned int);

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile global int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                int val) {
  if (semantics == Release) {
    __clc__atomic_store_global_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_global_4_seq_cst(p, val);
  } else {
    __clc__atomic_store_global_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile local int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                int val) {
  if (semantics == Release) {
    __clc__atomic_store_local_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_local_4_seq_cst(p, val);
  } else {
    __clc__atomic_store_local_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile global unsigned int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned int val) {
  if (semantics == Release) {
    __clc__atomic_ustore_global_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_global_4_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_global_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile local unsigned int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned int val) {
  if (semantics == Release) {
    __clc__atomic_ustore_local_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_local_4_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_local_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(global int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                int val) {
  if (semantics == Release) {
    __clc__atomic_store_global_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_global_4_seq_cst(p, val);
  } else {
    __clc__atomic_store_global_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(local int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                int val) {
  if (semantics == Release) {
    __clc__atomic_store_local_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_local_4_seq_cst(p, val);
  } else {
    __clc__atomic_store_local_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(global unsigned int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned int val) {
  if (semantics == Release) {
    __clc__atomic_ustore_global_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_global_4_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_global_4_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(local unsigned int *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned int val) {
  if (semantics == Release) {
    __clc__atomic_ustore_local_4_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_local_4_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_local_4_unordered(p, val);
  }
}

#ifdef cl_khr_int64_base_atomics
_CLC_DEF long __clc__atomic_store_global_8_unordered(global const long *, long);
_CLC_DEF long __clc__atomic_store_global_8_release(global const long *, long);
_CLC_DEF long __clc__atomic_store_global_8_seq_cst(global const long *, long);
_CLC_DEF long __clc__atomic_store_local_8_unordered(local const long *, long);
_CLC_DEF long __clc__atomic_store_local_8_release(local const long *, long);
_CLC_DEF long __clc__atomic_store_local_8_seq_cst(local const long *, long);
_CLC_DEF unsigned long
__clc__atomic_ustore_global_8_unordered(global const unsigned long *,
                                        unsigned long);
_CLC_DEF unsigned long
__clc__atomic_ustore_global_8_release(global const unsigned long *,
                                      unsigned long);
_CLC_DEF unsigned long
__clc__atomic_ustore_global_8_seq_cst(global const unsigned long *,
                                      unsigned long);
_CLC_DEF unsigned long
__clc__atomic_ustore_local_8_unordered(local const unsigned long *,
                                       unsigned long);
_CLC_DEF unsigned long
__clc__atomic_ustore_local_8_release(local const unsigned long *,
                                     unsigned long);
_CLC_DEF unsigned long
__clc__atomic_ustore_local_8_seq_cst(local const unsigned long *,
                                     unsigned long);

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile global long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                long val) {
  if (semantics == Release) {
    __clc__atomic_store_global_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_global_8_seq_cst(p, val);
  } else {
    __clc__atomic_store_global_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile local long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                long val) {
  if (semantics == Release) {
    __clc__atomic_store_local_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_local_8_seq_cst(p, val);
  } else {
    __clc__atomic_store_local_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_AtomicStore(volatile global unsigned long *p, unsigned int scope,
                    unsigned int semantics, unsigned long val) {
  if (semantics == Release) {
    __clc__atomic_ustore_global_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_global_8_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_global_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(volatile local unsigned long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned long val) {
  if (semantics == Release) {
    __clc__atomic_ustore_local_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_local_8_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_local_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(global long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                long val) {
  if (semantics == Release) {
    __clc__atomic_store_global_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_global_8_seq_cst(p, val);
  } else {
    __clc__atomic_store_global_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(local long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                long val) {
  if (semantics == Release) {
    __clc__atomic_store_local_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_store_local_8_seq_cst(p, val);
  } else {
    __clc__atomic_store_local_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(global unsigned long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned long val) {
  if (semantics == Release) {
    __clc__atomic_ustore_global_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_global_8_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_global_8_unordered(p, val);
  }
}

_CLC_OVERLOAD _CLC_DEF void __spirv_AtomicStore(local unsigned long *p,
                                                unsigned int scope,
                                                unsigned int semantics,
                                                unsigned long val) {
  if (semantics == Release) {
    __clc__atomic_ustore_local_8_release(p, val);
  } else if (semantics == SequentiallyConsistent) {
    __clc__atomic_ustore_local_8_seq_cst(p, val);
  } else {
    __clc__atomic_ustore_local_8_unordered(p, val);
  }
}
#endif // cl_khr_int64_base_atomics
