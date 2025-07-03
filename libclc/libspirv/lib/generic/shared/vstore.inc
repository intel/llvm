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

#define CLC_VSTORE_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)
#define CLC_VSTORE_NAME(x, y) __CLC_XCONCAT(__CLC_XCONCAT(x, vstore), y)
#define SPIRV_VSTORE_HALF_NAME(y)                                              \
  __CLC_XCONCAT(__CLC_XCONCAT(__spirv_ocl_vstore_half, N_IF_VEC), y)
#define CLC_VSTORE_HALF_NAME(y)                                                \
  __CLC_XCONCAT(__CLC_XCONCAT(__clc_vstore_half, __CLC_VECSIZE), y)

#define SPIRV_VSTOREA_HALF_NAME(y)                                             \
  __CLC_XCONCAT(__CLC_XCONCAT(__spirv_ocl_vstorea_half, N_IF_VEC), y)
#define CLC_VSTOREA_HALF_NAME(y)                                               \
  __CLC_XCONCAT(__CLC_XCONCAT(__clc_vstorea_half, __CLC_VECSIZE), y)

#ifdef __CLC_SCALAR

#define CLC_VSTORE_DEF(ADDRSPACE)                                              \
  _CLC_OVERLOAD _CLC_DEF void CLC_VSTORE_NAME(__spirv_ocl_, )(                 \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE __CLC_SCALAR_GENTYPE *p) {  \
    p[offset] = data;                                                          \
  }

#else

#define CLC_VSTORE_DEF(ADDRSPACE)                                              \
  _CLC_OVERLOAD _CLC_DEF void CLC_VSTORE_NAME(__spirv_ocl_, n)(                \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE __CLC_SCALAR_GENTYPE *p) {  \
    return CLC_VSTORE_NAME(__clc_, __CLC_VECSIZE)(data, offset, p);            \
  }

#endif

CLC_VSTORE_DEF(__private)
CLC_VSTORE_DEF(__local)
CLC_VSTORE_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VSTORE_DEF(__generic)
#endif

#undef CLC_VSTORE_DEF

// vstore_half and vstorea_half are available even if cl_khr_fp16 is
// unavailable.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64

#define CLC_VSTORE_HALF_DEF(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF void SPIRV_VSTORE_HALF_NAME()(                        \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                  \
    CLC_VSTORE_HALF_NAME()(data, offset, p);                                   \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF void SPIRV_VSTOREA_HALF_NAME()(                       \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                  \
    CLC_VSTOREA_HALF_NAME()(data, offset, p);                                  \
  }

#define CLC_VSTORE_HALF_RND_DEF(ADDRSPACE)                                     \
  _CLC_OVERLOAD _CLC_DEF void SPIRV_VSTORE_HALF_NAME(_r)(                      \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p,                    \
      unsigned int round_mode) {                                               \
    switch (round_mode) {                                                      \
    case SPV_RTE:                                                              \
      CLC_VSTORE_HALF_NAME(_rte)(data, offset, p);                             \
      break;                                                                   \
    case SPV_RTZ:                                                              \
      CLC_VSTORE_HALF_NAME(_rtz)(data, offset, p);                             \
      break;                                                                   \
    case SPV_RTP:                                                              \
      CLC_VSTORE_HALF_NAME(_rtp)(data, offset, p);                             \
      break;                                                                   \
    case SPV_RTN:                                                              \
      CLC_VSTORE_HALF_NAME(_rtn)(data, offset, p);                             \
      break;                                                                   \
    }                                                                          \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF void SPIRV_VSTOREA_HALF_NAME(_r)(                     \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p,                    \
      unsigned int round_mode) {                                               \
    switch (round_mode) {                                                      \
    case SPV_RTE:                                                              \
      CLC_VSTOREA_HALF_NAME(_rte)(data, offset, p);                            \
      break;                                                                   \
    case SPV_RTZ:                                                              \
      CLC_VSTOREA_HALF_NAME(_rtz)(data, offset, p);                            \
      break;                                                                   \
    case SPV_RTP:                                                              \
      CLC_VSTOREA_HALF_NAME(_rtp)(data, offset, p);                            \
      break;                                                                   \
    case SPV_RTN:                                                              \
      CLC_VSTOREA_HALF_NAME(_rtn)(data, offset, p);                            \
      break;                                                                   \
    }                                                                          \
  }

#define CLC_VSTORE_HALF_DEF_ALL_MODES(ADDRSPACE)                               \
  CLC_VSTORE_HALF_DEF(ADDRSPACE)                                               \
  CLC_VSTORE_HALF_RND_DEF(ADDRSPACE)

CLC_VSTORE_HALF_DEF_ALL_MODES(__private)
CLC_VSTORE_HALF_DEF_ALL_MODES(__local)
CLC_VSTORE_HALF_DEF_ALL_MODES(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VSTORE_HALF_DEF_ALL_MODES(__generic)
#endif

#undef CLC_VSTORE_HALF_DEF
#undef CLC_VSTORE_HALF_RND_DEF
#undef CLC_VSTORE_HALF_DEF_ALL_MODES

#endif
#endif

#undef N_IF_VEC
#undef CLC_VSTORE_TY
#undef CLC_VSTORE_NAME
#undef CLC_VSTORE_HALF_NAME
#undef CLC_VSTOREA_HALF_NAME
#undef SPIRV_VSTORE_HALF_NAME
#undef SPIRV_VSTOREA_HALF_NAME
