#ifndef CLC_SPIRV_IMAGE_COMMON
#define CLC_SPIRV_IMAGE_COMMON

#include <clc/clc_as_type.h>
#include <clc/opencl/clc.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef _WIN32
#define _CLC_MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)         \
  _Z##namelength##name##prefix##y##postfix
#else
#define _CLC_MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)         \
  _Z##namelength##name##prefix##m##postfix
#endif

// The ockl functions/builtins we link against from the ROCm device libs expect
// resources to reside in constant address space.
#if __clang_major__ >= 8
#define _CLC_CONST_AS __constant
#elif __clang_major__ >= 7
#define _CLC_CONST_AS __attribute__((address_space(4)))
#else
#define _CLC_CONST_AS __attribute__((address_space(2)))
#endif

// From
// https://github.com/ROCm/clr/tree/amd-staging/hipamd/include/hip/amd_detail/texture_fetch_functions.h
// defined in "image_common.cl"
extern _CLC_CONST_AS const unsigned int SAMPLER_OBJECT_OFFSET_DWORD;

// Helpers for casting between two builtin vector types and/or scalar types.

// Using __clc_as_type() and __clc_as_typen() functions to reinterpret types.
// The restriction being is that element "type"s need to be of the same size.
#define _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_VEC3(vec4_elem_t, to_t)              \
  _CLC_DECL to_t##3 __clc_cast_from_##vec4_elem_t##4_to_##to_t##3(             \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_VEC2(vec4_elem_t, to_t)              \
  _CLC_DECL to_t##2 __clc_cast_from_##vec4_elem_t##4_to_##to_t##2(             \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_SCALAR(vec4_elem_t, to_t)            \
  _CLC_DECL to_t __clc_cast_from_##vec4_elem_t##4_to_##to_t(                   \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_BUILTIN_CAST_VEC3_TO_VEC4(from_t, vec4_elem_t)            \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##3_to_##vec4_elem_t##4(    \
      from_t##3 from);

#define _CLC_DECLARE_BUILTIN_CAST_VEC2_TO_VEC4(from_t, vec4_elem_t)            \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##2_to_##vec4_elem_t##4(    \
      from_t##2 from);

#define _CLC_DECLARE_BUILTIN_CAST_SCALAR_TO_VEC4(from_t, vec4_elem_t)          \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##_to_##vec4_elem_t##4(     \
      from_t from);

// Generic casts between builtin types.
#define _CLC_DECLARE_CAST_VEC4(vec4_elem_t, to_t)                              \
  _CLC_DECL to_t##4 __clc_cast_from_##vec4_elem_t##4_to_##to_t##4(             \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_CAST_VEC4_TO_VEC3(vec4_elem_t, to_t)                      \
  _CLC_DECL to_t##3 __clc_cast_from_##vec4_elem_t##4_to_##to_t##3(             \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_CAST_VEC4_TO_VEC2(vec4_elem_t, to_t)                      \
  _CLC_DECL to_t##2 __clc_cast_from_##vec4_elem_t##4_to_##to_t##2(             \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_CAST_VEC4_TO_SCALAR(vec4_elem_t, to_t)                    \
  _CLC_DECL to_t __clc_cast_from_##vec4_elem_t##4_to_##to_t(                   \
      vec4_elem_t##4 from);

#define _CLC_DECLARE_CAST_VEC3_TO_VEC4(from_t, vec4_elem_t)                    \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##3_to_##vec4_elem_t##4(    \
      from_t##3 from);

#define _CLC_DECLARE_CAST_VEC2_TO_VEC4(from_t, vec4_elem_t)                    \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##2_to_##vec4_elem_t##4(    \
      from_t##2 from);

#define _CLC_DECLARE_CAST_SCALAR_TO_VEC4(from_t, vec4_elem_t)                  \
  _CLC_DECL vec4_elem_t##4 __clc_cast_from_##from_t##_to_##vec4_elem_t##4(     \
      from_t from);

// Helpers to extract N channel(s) from a four-channel (RGBA/XYZW) color type.

#define _CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(from_t, to_t)             \
  _CLC_DECLARE_CAST_VEC4(from_t, to_t)                                         \
  _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_VEC3(from_t, to_t)                         \
  _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_VEC2(from_t, to_t)                         \
  _CLC_DECLARE_BUILTIN_CAST_VEC4_TO_SCALAR(from_t, to_t)                       \
  _CLC_DECLARE_BUILTIN_CAST_VEC2_TO_VEC4(from_t, to_t)                         \
  _CLC_DECLARE_BUILTIN_CAST_VEC3_TO_VEC4(from_t, to_t)                         \
  _CLC_DECLARE_BUILTIN_CAST_SCALAR_TO_VEC4(from_t, to_t)

#define _CLC_DECLARE_EXTRACT_COLOR_HELPERS(from_t, to_t)                       \
  _CLC_DECLARE_CAST_VEC4(from_t, to_t)                                         \
  _CLC_DECLARE_CAST_VEC4_TO_VEC3(from_t, to_t)                                 \
  _CLC_DECLARE_CAST_VEC4_TO_VEC2(from_t, to_t)                                 \
  _CLC_DECLARE_CAST_VEC4_TO_SCALAR(from_t, to_t)                               \
  _CLC_DECLARE_CAST_VEC2_TO_VEC4(from_t, to_t)                                 \
  _CLC_DECLARE_CAST_VEC3_TO_VEC4(from_t, to_t)                                 \
  _CLC_DECLARE_CAST_SCALAR_TO_VEC4(from_t, to_t)

// Define casts between supported builtin types for image color

_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(float, float)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(float, int)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(int, float)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(float, uint)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(uint, float)
#ifdef cl_khr_fp16
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(half, half)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(half, short)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(short, half)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(half, ushort)
_CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS(ushort, half)
#endif

_CLC_DECLARE_EXTRACT_COLOR_HELPERS(half, float)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(float, short)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(short, float)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(float, ushort)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(ushort, float)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(float, char)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(char, float)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(float, uchar)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(uchar, float)
#ifdef cl_khr_fp16
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(float, half)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(half, int)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(int, half)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(half, uint)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(uint, half)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(half, char)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(char, half)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(half, uchar)
_CLC_DECLARE_EXTRACT_COLOR_HELPERS(uchar, half)
#endif

#undef _CLC_DECLARE_EXTRACT_COLOR_HELPERS
#undef _CLC_DECLARE_EXTRACT_SAME_SIZE_COLOR_HELPERS

#undef _CLC_DECLARE_CAST_SCALAR_TO_VEC4
#undef _CLC_DECLARE_CAST_VEC2_TO_VEC4
#undef _CLC_DECLARE_CAST_VEC3_TO_VEC4
#undef _CLC_DECLARE_CAST_VEC4_TO_SCALAR
#undef _CLC_DECLARE_CAST_VEC4_TO_VEC3
#undef _CLC_DECLARE_CAST_VEC4_TO_VEC2
#undef _CLC_DECLARE_CAST_VEC4
#undef _CLC_DECLARE_BUILTIN_CAST_SCALAR_TO_VEC4
#undef _CLC_DECLARE_BUILTIN_CAST_VEC2_TO_VEC4
#undef _CLC_DECLARE_BUILTIN_VEC3_TO_VEC4
#undef _CLC_DECLARE_BUILTIN_VEC4_TO_SCALAR
#undef _CLC_DECLARE_BUILTIN_VEC4_TO_VEC3
#undef _CLC_DECLARE_BUILTIN_VEC4_TO_VEC2
#undef _CLC_DECLARE_BUILTIN_VEC4

#endif // CLC_SPIRV_IMAGE_COMMON
