#ifndef SIMPLE_SYCL_LIB_H
#define SIMPLE_SYCL_LIB_H

#ifdef _WIN32
#define EXPORTDECL extern "C" __declspec(dllexport)
#else
#define EXPORTDECL extern "C"
#endif

EXPORTDECL int add_using_device(int a, int b);

#endif
