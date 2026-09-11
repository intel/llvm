// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// The pointer form of raw_kernel_arg takes the address of the pointer, in
// keeping with the byte form taking the address of the bytes. Passing the
// pointer itself would name an address the runtime would then read a pointer
// from, which is why it does not compile.

#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>

#include <type_traits>

namespace oneapiext = sycl::ext::oneapi::experimental;

// The graph extension copies a raw_kernel_arg as bytes.
static_assert(std::is_trivially_copyable_v<oneapiext::raw_kernel_arg>);

void pointer_form(int *Ptr, const float *ConstPtr, void *VoidPtr) {
  // Every pointer type binds through the address form without a cast.
  oneapiext::raw_kernel_arg Typed{&Ptr, oneapiext::pointer_arg};
  oneapiext::raw_kernel_arg Const{&ConstPtr, oneapiext::pointer_arg};
  oneapiext::raw_kernel_arg Void{&VoidPtr, oneapiext::pointer_arg};

  // The byte form is unchanged, including for the bytes of a pointer, which
  // only bind as a pointer on the Level Zero backend.
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
