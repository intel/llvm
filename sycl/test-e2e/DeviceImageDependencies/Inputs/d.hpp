#include <sycl/detail/core.hpp>

#if defined(MAKE_DLL)
#ifdef D_EXPORT
#define D_DECLSPEC __declspec(dllexport)
#else
#define D_DECLSPEC __declspec(dllimport)
#endif
#else
#define D_DECLSPEC
#endif

D_DECLSPEC SYCL_EXTERNAL int levelD(int val);
