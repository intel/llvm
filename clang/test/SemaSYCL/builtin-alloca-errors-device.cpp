// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -triple spir64-unknown-unknown -verify %s

// Test errors of __builtin_intel_sycl_alloca when used in SYCL device code.

#include <stddef.h>

#include "Inputs/sycl.hpp"
#include "Inputs/private_alloca.hpp"

constexpr sycl::specialization_id<size_t> size(1);
constexpr sycl::specialization_id<float> badsize(1);

struct wrapped_int { int a; };

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_0();

__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<float, sycl::access::address_space::private_space, sycl::access::decorated::no>
private_alloca_bad_1(sycl::kernel_handler &h);

template <typename ElementType>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, sycl::access::decorated::no>
private_alloca_bad_2(sycl::kernel_handler &h);

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_3(const wrapped_int &);

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_4(sycl::kernel_handler);

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_5(const sycl::kernel_handler &);

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::local_space, DecorateAddress>
private_alloca_bad_6(sycl::kernel_handler &);

template <typename ElementType, typename Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_7(sycl::kernel_handler &);

// expected-error@+4 {{cannot redeclare builtin function 'private_alloca'}}
// expected-note@+3 {{'private_alloca<float, size, sycl::access::decorated::no>' is a builtin with type 'multi_ptr<float, access::address_space::private_space, (decorated)0> (kernel_handler &)'}}
template <>
sycl::multi_ptr<float, sycl::access::address_space::private_space, sycl::access::decorated::no>
sycl::ext::oneapi::experimental::private_alloca<float, size, sycl::access::decorated::no>(sycl::kernel_handler &h);

void test(sycl::kernel_handler &h) {
  // expected-error@+1 {{builtin functions must be directly called}}
  auto funcPtr = sycl::ext::oneapi::experimental::private_alloca<float, size, sycl::access::decorated::no>;

  // expected-error@+1 {{__builtin_intel_sycl_alloca cannot be used in source code. Use the private_alloca alias instead}}
  __builtin_intel_sycl_alloca(h);

  // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
  private_alloca_bad_0<int, size, sycl::access::decorated::yes>();

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed three template arguments. Got 0}}
  private_alloca_bad_1(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed three template arguments. Got 1}}
  private_alloca_bad_2<float>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const wrapped_int &'}}
  private_alloca_bad_3<float, size, sycl::access::decorated::no>(wrapped_int{10});

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'sycl::kernel_handler'}}
  private_alloca_bad_4<float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const sycl::kernel_handler &'}}
  private_alloca_bad_5<float, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'multi_ptr<const float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<const float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'multi_ptr<volatile float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<volatile float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'multi_ptr<void, access::address_space::private_space, (decorated)1>'}}
  sycl::ext::oneapi::experimental::private_alloca<void, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'multi_ptr<int *(int), access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<int *(int), size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'multi_ptr<int &, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<int &, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified object type. Got 'sycl::multi_ptr<float, sycl::access::address_space::local_space, (decorated)0>'}}
  private_alloca_bad_6<float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca must be passed a specialization constant of integral value type as a template argument. Got 'int'}}
  private_alloca_bad_7<float, int, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca must be passed a specialization constant of integral value type as a template argument. Got 'const sycl::specialization_id<float> &'}}
  sycl::ext::oneapi::experimental::private_alloca<float, badsize, sycl::access::decorated::yes>(h);
}
