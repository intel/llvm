// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -DUSE_MINIMAL_INCLUDES %s

// Regression test: SYCL_EXT_ONEAPI_FUNCTION_PROPERTY must accept grf_size and
// grf_size_automatic kernel properties.

#ifdef USE_MINIMAL_INCLUDES
// Verify the lightweight include path works without pulling in sycl.hpp.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#else
#include <sycl/sycl.hpp>
#endif

namespace intelex = sycl::ext::intel::experimental;
namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelex::grf_size<128>))
void kernel_grf128() {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelex::grf_size<256>))
void kernel_grf256() {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelex::grf_size_automatic))
void kernel_grf_automatic() {}
