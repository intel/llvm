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
// TODO: Add expected error when it will be implemented.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelVariadic(...) {}

// TODO: Add expected error when it will be implemented.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<3>)
void ndRangeKernelVariadic(...) {}

// expected-error@+2 {{a function with a default argument value cannot be used as a kernel function, 'int Value = 1'}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelDefaultValues(int Value = 1) {}

// expected-error@+2 {{a function with a default argument value cannot be used as a kernel function, 'int Value = 1'}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelDefaultValues(int Value = 1) {}
