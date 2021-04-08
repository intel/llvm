#ifndef CLC_MACROS
#define CLC_MACROS

/* 6.9 Preprocessor Directives and Macros
 * Some of these are handled by clang or passed by clover */
#if __OPENCL_VERSION__ >= 110
#define CLC_VERSION_1_0 100
#define CLC_VERSION_1_1 110
#endif

#if __OPENCL_VERSION__ >= 120
#define CLC_VERSION_1_2 120
#endif

#define NULL ((void*)0)

#endif // CLC_MACROS
