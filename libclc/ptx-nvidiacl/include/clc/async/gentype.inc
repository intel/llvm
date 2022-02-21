
#define __CLC_GENTYPE char
#define __CLC_GENTYPE_MANGLED c
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char2
#define __CLC_GENTYPE_MANGLED Dv2_c
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE char3
#define __CLC_GENTYPE_U uchar3
#define __CLC_GENTYPE_MANGLED Dv3_c
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE_U
#undef __CLC_GENTYPE
#endif

#ifndef __CLC_NO_SCHAR
#define __CLC_GENTYPE schar
#define __CLC_GENTYPE_MANGLED a
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE schar2
#define __CLC_GENTYPE_MANGLED Dv2_a
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE schar3
#define __CLC_GENTYPE_MANGLED Dv3_a
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#endif

#define __CLC_GENTYPE uchar
#define __CLC_GENTYPE_MANGLED h
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar2
#define __CLC_GENTYPE_MANGLED Dv2_h
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE uchar3
#define __CLC_GENTYPE_MANGLED Dv3_h
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE short
#define __CLC_GENTYPE_MANGLED s
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE short3
#define __CLC_GENTYPE_MANGLED Dv3_s
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE short16
#define __CLC_GENTYPE_MANGLED Dv16_s
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort
#define __CLC_GENTYPE_MANGLED t
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE ushort3
#define __CLC_GENTYPE_MANGLED Dv3_t
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE ushort16
#define __CLC_GENTYPE_MANGLED Dv16_t
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE int3
#define __CLC_GENTYPE_MANGLED Dv3_i
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE int8
#define __CLC_GENTYPE_MANGLED Dv8_i
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int16
#define __CLC_GENTYPE_MANGLED Dv16_i
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE uint3
#define __CLC_GENTYPE_MANGLED Dv3_j
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE uint8
#define __CLC_GENTYPE_MANGLED Dv8_j
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint16
#define __CLC_GENTYPE_MANGLED Dv16_j
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE float3
#define __CLC_GENTYPE_MANGLED Dv3_f
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE float8
#define __CLC_GENTYPE_MANGLED Dv8_f
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE float16
#define __CLC_GENTYPE_MANGLED Dv16_f
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE long3
#define __CLC_GENTYPE_MANGLED Dv3_l
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE long4
#define __CLC_GENTYPE_MANGLED Dv4_l
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long8
#define __CLC_GENTYPE_MANGLED Dv8_l
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long16
#define __CLC_GENTYPE_MANGLED Dv16_l
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE ulong3
#define __CLC_GENTYPE_MANGLED Dv3_m
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE ulong4
#define __CLC_GENTYPE_MANGLED Dv4_m
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong8
#define __CLC_GENTYPE_MANGLED Dv8_m
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong16
#define __CLC_GENTYPE_MANGLED Dv16_m
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE double3
#define __CLC_GENTYPE_MANGLED Dv3_d
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE double4
#define __CLC_GENTYPE_MANGLED Dv4_d
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double8
#define __CLC_GENTYPE_MANGLED Dv8_d
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#define __CLC_GENTYPE double16
#define __CLC_GENTYPE_MANGLED Dv16_d
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16: enable

#define __CLC_GENTYPE half
#define __CLC_GENTYPE_MANGLED Dh
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#ifdef __CLC_GEN_VEC3
#define __CLC_GENTYPE half3
#define __CLC_GENTYPE_MANGLED Dv3_h
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE
#endif

#define __CLC_GENTYPE half16
#define __CLC_GENTYPE_MANGLED Dv16_Dh
#include __CLC_BODY
#undef __CLC_GENTYPE_MANGLED
#undef __CLC_GENTYPE

#endif

#undef __CLC_BODY
