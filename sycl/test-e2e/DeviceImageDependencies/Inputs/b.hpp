#include <sycl/detail/core.hpp>

#if defined(MAKE_DLL)
#ifdef B_EXPORT
#define B_DECLSPEC __declspec(dllexport)
#else
#define B_DECLSPEC __declspec(dllimport)
#endif
#else
#define B_DECLSPEC
#endif

B_DECLSPEC SYCL_EXTERNAL int levelB(int val);
