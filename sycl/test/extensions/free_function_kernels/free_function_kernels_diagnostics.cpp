// RUN: %clangxx -fsyntax-only -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// expected-error@+2 {{'int &' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelReference(int &Ref) {}

// expected-error@+2 {{'int &' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<2>)
void ndRangeKernelReference(int &Ref) {}

// Diagnostic for these violations of the restrictions haven't been implemented
// yet.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<3>)
void ndRangeKernelVariadic( // expected-error {{free function kernel cannot be a variadic function}}
    ...) {}
