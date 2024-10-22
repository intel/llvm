#pragma once
#include <sycl/sycl.hpp>

#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#define __DEVICE_CODE 1
#else
#define __DEVICE_CODE 0
#endif
