#ifndef CLC_AS_TYPE
#define CLC_AS_TYPE

#define as_schar(x) __builtin_astype(x, schar)
#define as_schar2(x) __builtin_astype(x, schar2)
#define as_schar3(x) __builtin_astype(x, schar3)
#define as_schar4(x) __builtin_astype(x, schar4)
#define as_schar8(x) __builtin_astype(x, schar8)
#define as_schar16(x) __builtin_astype(x, schar16)

#ifdef __CLC_HAS_FLOAT16
#define as_float16_t(x) __builtin_astype(x, __clc_float16_t)
#define as_vec2_float16_t(x) __builtin_astype(x, __clc_vec2_float16_t)
#define as_vec3_float16_t(x) __builtin_astype(x, __clc_vec3_float16_t)
#define as_vec4_float16_t(x) __builtin_astype(x, __clc_vec4_float16_t)
#define as_vec8_float16_t(x) __builtin_astype(x, __clc_vec8_float16_t)
#define as_vec16_float16_t(x) __builtin_astype(x, __clc_vec16_float16_t)
#endif

#endif // CLC_AS_TYPE
