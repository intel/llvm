// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// The pointer_arg constructor takes the address of a pointer, not its value.

#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>

#include <type_traits>

namespace oneapiext = sycl::ext::oneapi::experimental;

// dynamic_parameter copies a raw_kernel_arg as bytes.
static_assert(std::is_trivially_copyable_v<oneapiext::raw_kernel_arg>);

void pointer_form(int *Ptr, const float *ConstPtr, void *VoidPtr) {
  // Pointers to const and to void are accepted without a cast too.
  oneapiext::raw_kernel_arg Typed{&Ptr, oneapiext::pointer_arg};
  oneapiext::raw_kernel_arg Const{&ConstPtr, oneapiext::pointer_arg};
  oneapiext::raw_kernel_arg Void{&VoidPtr, oneapiext::pointer_arg};

  // An address and a size still select the byte form.
  oneapiext::raw_kernel_arg Bytes{&Ptr, sizeof(Ptr)};

  // expected-error@+1 {{no matching constructor for initialization of 'oneapiext::raw_kernel_arg'}}
  oneapiext::raw_kernel_arg PointerItself{Ptr, oneapiext::pointer_arg};

  // expected-error@+1 {{no matching constructor for initialization of 'oneapiext::raw_kernel_arg'}}
  oneapiext::raw_kernel_arg NotAPointer{42, oneapiext::pointer_arg};

  (void)Typed;
  (void)Const;
  (void)Void;
  (void)Bytes;
}
