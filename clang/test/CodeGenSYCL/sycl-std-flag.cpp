// RUN: %clang++  -fsycl  -fsyntax-only -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang++  -fsycl  -fsyntax-only -sycl-std=121 -DSYCL2017 %s
// RUN: %clang++  -fsycl  -fsyntax-only -sycl-std=1.2.1 -DSYCL2017 %s
// RUN: %clang++  -fsycl  -fsyntax-only -sycl-std=sycl-1.2.1 -DSYCL2017 %s
// RUN: %clang++  -fsycl  -fsyntax-only -sycl-std=2020 -DSYCL2020 %s


#if defined(SYCL2017)
    #if !CL_SYCL_LANGUAGE_VERSION
        static_assert(false , "CL_SYCL_LANGUAGE_VERSION should be defined when -sycl-std flag is used");
    #endif
    #if  CL_SYCL_LANGUAGE_VERSION > 121
        static_assert(false, "-sycl-std flag should have set CL_SYCL_LANGUAGE_VERSION to 121");
    #endif
#endif

#if defined(SYCL2020)
    #if !CL_SYCL_LANGUAGE_VERSION
        static_assert(false , "CL_SYCL_LANGUAGE_VERSION should be defined when -sycl-std flag is used");
    #endif
    #if  CL_SYCL_LANGUAGE_VERSION < 2020
        static_assert(false , "-sycl-std=2020 flag should set CL_SYCL_LANGUAGE_VERSION to 2020");
    #endif
#endif

// expected-no-diagnostics