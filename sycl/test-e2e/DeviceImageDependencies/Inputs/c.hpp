#include <sycl/detail/core.hpp>

#if defined(MAKE_DLL)
#ifdef C_EXPORT
#define C_DECLSPEC __declspec(dllexport)
#else
#define C_DECLSPEC __declspec(dllimport)
#endif
#else
#define C_DECLSPEC
#endif

C_DECLSPEC SYCL_EXTERNAL int levelC(int val);
