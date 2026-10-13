// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// This test checks that the free function kernel enqueue functions are
// constrained on the arguments being usable to call the kernel, as the
// specification of the free function kernels extension requires, so that
// argument lists which the kernel cannot accept are diagnosed at the call site
// instead of setting bogus kernel arguments.

#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct NotAnInt {};

// A user-defined type which implicitly converts to the kernel's parameter type.
struct ConvertibleToInt {
  int Value;
  operator int() const { return Value; }
};

// The same conversion, declared `explicit`. This does not make the type
// implicitly convertible, so it does not satisfy `is_invocable_v` either and
// the call is rejected, even though the `static_cast` the enqueue functions
// perform internally could apply the conversion. That matches calling the
// kernel directly, which is what the argument conversion mirrors.
struct ExplicitConvertibleToInt {
  int Value;
  explicit operator int() const { return Value; }
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void singleTaskKernel(int Factor, int *Ptr) { *Ptr = *Ptr * Factor; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void ndRangeKernel(int Factor, int *Ptr) { *Ptr = *Ptr * Factor; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void singleTaskFloatKernel(float Factor, float *Ptr) { *Ptr = *Ptr * Factor; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void ndRangeFloatKernel(float Factor, float *Ptr) { *Ptr = *Ptr * Factor; }

void test(sycl::queue Q, sycl::handler &CGH, int *Ptr, float *FPtr) {
  const sycl::nd_range<1> Range{sycl::range<1>(1), sycl::range<1>(1)};
  const syclexp::launch_config Config{Range};

  // Arguments which convert to the kernel's parameter types are accepted.
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>, 2.0, Ptr);
  syclexp::single_task(CGH, syclexp::kernel_function<singleTaskKernel>, 2.0,
                       Ptr);
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>, 2.0,
                     Ptr);
  syclexp::nd_launch(CGH, Range, syclexp::kernel_function<ndRangeKernel>, 2.0,
                     Ptr);
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeKernel>, 2.0,
                     Ptr);
  syclexp::nd_launch(CGH, Config, syclexp::kernel_function<ndRangeKernel>, 2.0,
                     Ptr);

  // A user-defined type with an implicit conversion operator is accepted too.
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>,
                       ConvertibleToInt{2}, Ptr);
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>,
                     ConvertibleToInt{2}, Ptr);
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeKernel>,
                     ConvertibleToInt{2}, Ptr);

  // The constraint does not reject narrowing conversions either, so a double
  // argument passed to a float parameter is accepted and converted.
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskFloatKernel>, 2.0,
                       FPtr);
  syclexp::single_task(CGH, syclexp::kernel_function<singleTaskFloatKernel>,
                       2.0, FPtr);
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeFloatKernel>,
                     2.0, FPtr);
  syclexp::nd_launch(CGH, Range, syclexp::kernel_function<ndRangeFloatKernel>,
                     2.0, FPtr);
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeFloatKernel>,
                     2.0, FPtr);
  syclexp::nd_launch(CGH, Config, syclexp::kernel_function<ndRangeFloatKernel>,
                     2.0, FPtr);

  // Too few arguments.
  // expected-error@+1 {{no matching function for call to 'single_task'}}
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>, 2);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>, 2);

  // Too many arguments.
  // expected-error@+1 {{no matching function for call to 'single_task'}}
  syclexp::single_task(CGH, syclexp::kernel_function<singleTaskKernel>, 2, Ptr,
                       3);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(CGH, Config, syclexp::kernel_function<ndRangeKernel>, 2,
                     Ptr, 3);

  // An argument which does not convert to the parameter type at all.
  // expected-error@+1 {{no matching function for call to 'single_task'}}
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>,
                       NotAnInt{}, Ptr);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>,
                     NotAnInt{}, Ptr);

  // An argument whose conversion operator is `explicit` does not convert
  // implicitly, so it is rejected as well.
  // expected-error@+1 {{no matching function for call to 'single_task'}}
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>,
                       ExplicitConvertibleToInt{2}, Ptr);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(Q, Range, syclexp::kernel_function<ndRangeKernel>,
                     ExplicitConvertibleToInt{2}, Ptr);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeKernel>,
                     ExplicitConvertibleToInt{2}, Ptr);

  // Pointers to a different type do not convert either, so a mismatch which
  // used to be passed on to the device is now diagnosed here.
  // expected-error@+1 {{no matching function for call to 'single_task'}}
  syclexp::single_task(Q, syclexp::kernel_function<singleTaskKernel>, 2, FPtr);
  // expected-error@+1 {{no matching function for call to 'nd_launch'}}
  syclexp::nd_launch(Q, Config, syclexp::kernel_function<ndRangeKernel>, 2,
                     FPtr);
}
