//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define GenericCastToPtrExplicit_To(ADDRSPACE, NAME)                           \
  _CLC_DECL _CLC_OVERLOAD                                                      \
      ADDRSPACE void *__spirv_GenericCastToPtrExplicit_To##NAME(               \
          generic void *, int);                                                \
  _CLC_DECL _CLC_OVERLOAD                                                      \
      ADDRSPACE const void *__spirv_GenericCastToPtrExplicit_To##NAME(         \
          generic const void *, int);                                          \
  _CLC_DECL _CLC_OVERLOAD                                                      \
      ADDRSPACE volatile void *__spirv_GenericCastToPtrExplicit_To##NAME(      \
          generic volatile void *, int);                                       \
  _CLC_DECL _CLC_OVERLOAD ADDRSPACE const volatile void *                      \
      __spirv_GenericCastToPtrExplicit_To##NAME(generic const volatile void *, \
                                                int)

GenericCastToPtrExplicit_To(global, Global);
GenericCastToPtrExplicit_To(local, Local);
GenericCastToPtrExplicit_To(private, Private);

#undef GenericCastToPtrExplicit_To
