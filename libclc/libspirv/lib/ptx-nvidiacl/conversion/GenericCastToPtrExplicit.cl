//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF static bool __clc_nvvm_is_private(generic void *ptr) {
  return __nvvm_isspacep_local(ptr);
}
_CLC_DEF static bool __clc_nvvm_is_local(generic void *ptr) {
  return __nvvm_isspacep_shared(ptr);
}
_CLC_DEF static bool __clc_nvvm_is_global(generic void *ptr) {
  return __nvvm_isspacep_global(ptr);
}

#define GenericCastToPtrExplicit_To(ADDRSPACE, NAME)                           \
  _CLC_DECL _CLC_OVERLOAD                                                      \
      ADDRSPACE void *__spirv_GenericCastToPtrExplicit_To##NAME(               \
          generic void *ptr, int unused) {                                     \
    if (__clc_nvvm_is_##ADDRSPACE(ptr))                                        \
      return (ADDRSPACE void *)ptr;                                            \
    return 0;                                                                  \
  }                                                                            \
  _CLC_DECL _CLC_OVERLOAD                                                      \
      ADDRSPACE const void *__spirv_GenericCastToPtrExplicit_To##NAME(         \
          generic const void *ptr, int unused) {                               \
    return __spirv_GenericCastToPtrExplicit_To##NAME((generic void *)ptr,      \
                                                     unused);                  \
  }

GenericCastToPtrExplicit_To(global, Global)
GenericCastToPtrExplicit_To(local, Local)
GenericCastToPtrExplicit_To(private, Private)
