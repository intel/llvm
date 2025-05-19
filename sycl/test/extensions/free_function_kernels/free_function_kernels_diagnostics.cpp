// RUN: %clangxx -fsyntax-only -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// expected-error@+2 {{'int &' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelReference(int &Ref) {}

// expected-error@+2 {{'int &' cannot be used as the type of a kernel parameter}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<2>)
void ndRangeKernelReference(int &Ref) {}

// expected-error@+2 {{a function with a default argument value cannot
// be used to define SYCL free function kernel}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelDefaultParameterValue(int DefVal = 1024) {}

// expected-error@+2 {{a function with a default argument value
// cannot be used to define SYCL free function kernel}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<3>)
void ndRangeKernelReferenceDefaultParameterValue(int DefVal = 1024) {}

// expected-error@+2 {{free function kernel cannot be a variadic function}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelVariadic(...) {}

// expected-error@+2 {{free function kernel cannot be a variadic function}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelVariadic(...) {}

class DummyClass {
public:
  // Diagnostic for these violations of the restrictions haven't been
  // implemented yet.
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
  void singleTaskKernelNonStaticMemberFunc(int *Ptr) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<2>)
  void ndRangeKernelNonStaticMemberFunc(float *Ptr) {}
};

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
float singleTaskKernelNonVoid() { return 0.0F; }

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<3>)
int ndRangeKernelNonVoid() { return 0; }
