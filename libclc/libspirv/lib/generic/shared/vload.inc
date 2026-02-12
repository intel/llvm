//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR
#define N_IF_VEC
#else
#define N_IF_VEC n
#endif

#define CLC_VLOAD_NAME() __CLC_XCONCAT(__clc_vload, __CLC_VECSIZE)
#define SPIRV_VLOAD_NAME()                                                     \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(__spirv_ocl_vload, N_IF_VEC), _R), \
                __CLC_GENTYPE)

#define CLC_VLOAD_HALF_NAME(a)                                                 \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(__clc_vload, a), _half),           \
                __CLC_VECSIZE)

#define CLC_VLOAD_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)

#ifdef __CLC_SCALAR

#define VLOAD_DEF(ADDRSPACE)                                                   \
  _CLC_OVERLOAD _CLC_DEF CLC_VLOAD_TY SPIRV_VLOAD_NAME()(                      \
      size_t offset, const ADDRSPACE __CLC_SCALAR_GENTYPE *x) {                \
    return *((const ADDRSPACE CLC_VLOAD_TY *)(&x[offset]));                    \
  }

#define SPIRV_VLOAD_HALF_NAME(a)                                               \
  __CLC_XCONCAT(__CLC_XCONCAT(__spirv_ocl_vload, a), _half)

#else

#define VLOAD_DEF(ADDRSPACE)                                                   \
  _CLC_OVERLOAD _CLC_DEF CLC_VLOAD_TY SPIRV_VLOAD_NAME()(                      \
      size_t offset, const ADDRSPACE __CLC_SCALAR_GENTYPE *x) {                \
    return CLC_VLOAD_NAME()(offset, x);                                        \
  }

#define SPIRV_VLOAD_HALF_NAME(a)                                               \
  __CLC_XCONCAT(                                                               \
      __CLC_XCONCAT(                                                           \
          __CLC_XCONCAT(__CLC_XCONCAT(__spirv_ocl_vload, a), _halfn), _R),     \
      __CLC_GENTYPE)

#endif

VLOAD_DEF(__private)
VLOAD_DEF(__local)
VLOAD_DEF(__constant)
VLOAD_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
VLOAD_DEF(__generic)
#endif

#undef VLOAD_DEF
#undef CLC_VLOAD_TY

// vload_half and vloada_half are available even if cl_khr_fp16 is unavailable.
// Declare these functions when working on float types, which we know are
// always available.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32

#define VLOAD_HALF_DEF(ADDRSPACE, A)                                           \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE SPIRV_VLOAD_HALF_NAME(A)(               \
      size_t offset, const ADDRSPACE half *mem) {                              \
    return CLC_VLOAD_HALF_NAME(A)(offset, mem);                                \
  }

VLOAD_HALF_DEF(__private, )
VLOAD_HALF_DEF(__local, )
VLOAD_HALF_DEF(__constant, )
VLOAD_HALF_DEF(__global, )

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
VLOAD_HALF_DEF(__generic, )
#endif

#ifndef __CLC_SCALAR
VLOAD_HALF_DEF(__private, a)
VLOAD_HALF_DEF(__local, a)
VLOAD_HALF_DEF(__constant, a)
VLOAD_HALF_DEF(__global, a)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
VLOAD_HALF_DEF(__generic, a)
#endif
#endif

#undef VLOAD_HALF_DEF
#endif
#endif

#undef CLC_VLOAD_NAME
#undef CLC_VLOAD_HALF_NAME
#undef SPIRV_VLOAD_NAME
#undef SPIRV_VLOAD_HALF_NAME

#undef N_IF_VEC
