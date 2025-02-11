#include <sycl/detail/core.hpp>

#if defined(MAKE_DLL)
#ifdef A_EXPORT
#define A_DECLSPEC __declspec(dllexport)
#else
#define A_DECLSPEC __declspec(dllimport)
#endif
#else
#define A_DECLSPEC
#endif

A_DECLSPEC SYCL_EXTERNAL int levelA(int val);
