#ifndef __MANGLE_COMMON
#define __MANGLE_COMMON

#define MANUALLY_MANGLED_V_V_VP_VECTORIZE(SCALAR_NAME, FUNCTION_MACRO,         \
                                          ARG1_TYPE, ADDR_SPACE, ARG2_TYPE)    \
  _CLC_DEF ARG1_TYPE##2 FUNCTION_MACRO(2)(                                     \
      ARG1_TYPE##2 x,                                                          \
      __attribute__((address_space(ADDR_SPACE))) ARG2_TYPE##2 * y) {           \
    return (ARG1_TYPE##2)(                                                     \
        SCALAR_NAME(                                                           \
            x.x, (__attribute__((address_space(ADDR_SPACE))) ARG2_TYPE *)y),   \
        SCALAR_NAME(                                                           \
            x.y,                                                               \
            (__attribute__((address_space(ADDR_SPACE))) ARG2_TYPE *)y + 1));   \
  }                                                                            \
                                                                               \
  _CLC_DEF ARG1_TYPE##3 FUNCTION_MACRO(3)(                                     \
      ARG1_TYPE##3 x,                                                          \
      __attribute__((address_space(ADDR_SPACE))) ARG2_TYPE##3 * y) {           \
    return (ARG1_TYPE##3)(                                                     \
        SCALAR_NAME(                                                           \
            x.x, (__attribute__((address_space(ADDR_SPACE))) ARG2_TYPE *)y),   \
        SCALAR_NAME(                                                           \
            x.y,                                                               \
            (__attribute__((address_space(ADDR_SPACE))) ARG2_TYPE *)y + 1),    \
        SCALAR_NAME(                                                           \
            x.z,                                                               \
            (__attribute__((address_space(ADDR_SPACE))) ARG2_TYPE *)y + 2));   \
  }                                                                            \
                                                                               \
  _CLC_DEF ARG1_TYPE##4 FUNCTION_MACRO(4)(                                     \
      ARG1_TYPE##4 x,                                                          \
      __attribute__((address_space(ADDR_SPACE))) ARG2_TYPE##4 * y) {           \
    return (ARG1_TYPE##4)(                                                     \
        FUNCTION_MACRO(2)(x.lo, (__attribute__((address_space(ADDR_SPACE)))    \
                                 ARG2_TYPE##2 *)y),                            \
        FUNCTION_MACRO(2)(                                                     \
            x.hi, (__attribute__((address_space(ADDR_SPACE)))                  \
                   ARG2_TYPE##2 *)((__attribute__((address_space(ADDR_SPACE))) \
                                    ARG2_TYPE *)y +                            \
                                   2)));                                       \
  }                                                                            \
                                                                               \
  _CLC_DEF ARG1_TYPE##8 FUNCTION_MACRO(8)(                                     \
      ARG1_TYPE##8 x,                                                          \
      __attribute__((address_space(ADDR_SPACE))) ARG2_TYPE##8 * y) {           \
    return (ARG1_TYPE##8)(                                                     \
        FUNCTION_MACRO(4)(x.lo, (__attribute__((address_space(ADDR_SPACE)))    \
                                 ARG2_TYPE##4 *)y),                            \
        FUNCTION_MACRO(4)(                                                     \
            x.hi, (__attribute__((address_space(ADDR_SPACE)))                  \
                   ARG2_TYPE##4 *)((__attribute__((address_space(ADDR_SPACE))) \
                                    ARG2_TYPE *)y +                            \
                                   4)));                                       \
  }                                                                            \
                                                                               \
  _CLC_DEF ARG1_TYPE##16 FUNCTION_MACRO(16)(                                   \
      ARG1_TYPE##16 x,                                                         \
      __attribute__((address_space(ADDR_SPACE))) ARG2_TYPE##16 * y) {          \
    return (ARG1_TYPE##16)(                                                    \
        FUNCTION_MACRO(8)(x.lo, (__attribute__((address_space(ADDR_SPACE)))    \
                                 ARG2_TYPE##8 *)y),                            \
        FUNCTION_MACRO(8)(                                                     \
            x.hi, (__attribute__((address_space(ADDR_SPACE)))                  \
                   ARG2_TYPE##8 *)((__attribute__((address_space(ADDR_SPACE))) \
                                    ARG2_TYPE *)y +                            \
                                   8)));                                       \
  }

#endif // !__MANGLE_COMMON
