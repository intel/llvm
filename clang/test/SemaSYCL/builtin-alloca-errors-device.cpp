// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -triple spir64-unknown-unknown -verify %s

// Test errors of __builtin_intel_sycl_alloca and
// __builtin_intel_sycl_alloca_with_align when used in SYCL device code.

#include <stddef.h>

#include "Inputs/sycl.hpp"
#include "Inputs/private_alloca.hpp"

constexpr int ten() { return 10; }

constexpr sycl::specialization_id<size_t> size(1);
constexpr sycl::specialization_id<float> badsize(1);
constexpr sycl::specialization_id<size_t> zero;
constexpr sycl::specialization_id<int> negative(-1);
constexpr sycl::specialization_id<int> negative_expr(1 - ten());

constexpr const sycl::specialization_id<int> &negative_expr_ref = negative_expr;

template <typename T>
constexpr T exp2(unsigned a) { return a == 0 ? 1 : 2 * exp2<T>(a - 1); }

struct wrapped_int { int a; };

struct non_trivial { int a = 1; };

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

template <typename ElementType, size_t Alignment, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
aligned_private_alloca_bad_0();

__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<float, sycl::access::address_space::private_space, sycl::access::decorated::no>
aligned_private_alloca_bad_1(sycl::kernel_handler &h);

template <typename ElementType>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, sycl::access::decorated::no>
aligned_private_alloca_bad_2(sycl::kernel_handler &h);

template <typename ElementType, size_t Alignment, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
aligned_private_alloca_bad_3(const wrapped_int &);

template <typename ElementType, size_t Alignment, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
aligned_private_alloca_bad_4(sycl::kernel_handler);

template <typename ElementType, size_t Alignment, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
aligned_private_alloca_bad_5(const sycl::kernel_handler &);

template <typename ElementType, size_t Alignment, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::local_space, DecorateAddress>
aligned_private_alloca_bad_6(sycl::kernel_handler &);

template <typename ElementType, size_t Alignment, typename Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
aligned_private_alloca_bad_7(sycl::kernel_handler &);

// expected-error@+4 {{cannot redeclare builtin function 'private_alloca'}}
// expected-note@+3 {{'private_alloca<float, size, sycl::access::decorated::no>' is a builtin with type 'multi_ptr<float, access::address_space::private_space, (decorated)0> (kernel_handler &)'}}
template <>
sycl::multi_ptr<float, sycl::access::address_space::private_space, sycl::access::decorated::no>
sycl::ext::oneapi::experimental::private_alloca<float, size, sycl::access::decorated::no>(sycl::kernel_handler &h);

// expected-error@+4 {{cannot redeclare builtin function 'aligned_private_alloca'}}
// expected-note@+3 {{'aligned_private_alloca<float, 8UL, size, sycl::access::decorated::no>' is a builtin with type 'multi_ptr<float, access::address_space::private_space, (decorated)0> (kernel_handler &)'}}
template <>
sycl::multi_ptr<float, sycl::access::address_space::private_space, sycl::access::decorated::no>
sycl::ext::oneapi::experimental::aligned_private_alloca<float, alignof(float) * 2, size, sycl::access::decorated::no>(sycl::kernel_handler &h);

void test(sycl::kernel_handler &h) {
  // expected-error@+1 {{builtin functions must be directly called}}
  auto funcPtr = sycl::ext::oneapi::experimental::private_alloca<float, size, sycl::access::decorated::no>;

  // expected-error@+1 {{__builtin_intel_sycl_alloca cannot be used in source code. Use the private_alloca alias instead}}
  __builtin_intel_sycl_alloca(h);

  // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
  private_alloca_bad_0<int, size, sycl::access::decorated::yes>();

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed 3 template arguments. Got 0}}
  private_alloca_bad_1(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed 3 template arguments. Got 1}}
  private_alloca_bad_2<float>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const wrapped_int &'}}
  private_alloca_bad_3<float, size, sycl::access::decorated::no>(wrapped_int{10});

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'sycl::kernel_handler'}}
  private_alloca_bad_4<float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const sycl::kernel_handler &'}}
  private_alloca_bad_5<float, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<const float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<const float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<volatile float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<volatile float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<void, access::address_space::private_space, (decorated)1>'}}
  sycl::ext::oneapi::experimental::private_alloca<void, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<int *(int), access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<int *(int), size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<int &, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::private_alloca<int &, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'sycl::multi_ptr<float, sycl::access::address_space::local_space, (decorated)0>'}}
  private_alloca_bad_6<float, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<non_trivial, access::address_space::private_space, (decorated)1>'}}
  sycl::ext::oneapi::experimental::private_alloca<non_trivial, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca must be passed a specialization constant of integral value type as a template argument. Got 'int'}}
  private_alloca_bad_7<float, int, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca must be passed a specialization constant of integral value type as a template argument. Got 'const sycl::specialization_id<float> &'}}
  sycl::ext::oneapi::experimental::private_alloca<float, badsize, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca expects a specialization constant with a default value of at least one as an argument. Got 0}}
  sycl::ext::oneapi::experimental::private_alloca<float, zero, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca expects a specialization constant with a default value of at least one as an argument. Got -1}}
  sycl::ext::oneapi::experimental::private_alloca<float, negative, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca expects a specialization constant with a default value of at least one as an argument. Got -9}}
  sycl::ext::oneapi::experimental::private_alloca<float, negative_expr_ref, sycl::access::decorated::yes>(h);

  constexpr size_t alignment = 16;

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align cannot be used in source code. Use the aligned_private_alloca alias instead}}
  __builtin_intel_sycl_alloca_with_align(h);

  // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
  aligned_private_alloca_bad_0<int, alignment, size, sycl::access::decorated::yes>();

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align expects to be passed 4 template arguments. Got 0}}
  aligned_private_alloca_bad_1(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align expects to be passed 4 template arguments. Got 1}}
  aligned_private_alloca_bad_2<float>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const wrapped_int &'}}
  aligned_private_alloca_bad_3<float, alignment, size, sycl::access::decorated::no>(wrapped_int{10});

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'sycl::kernel_handler'}}
  aligned_private_alloca_bad_4<float, alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align expects to be passed an argument of type 'sycl::kernel_handler &'. Got 'const sycl::kernel_handler &'}}
  aligned_private_alloca_bad_5<float, alignment, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<const float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<const float, alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<volatile float, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<volatile float, alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<void, access::address_space::private_space, (decorated)1>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<void, alignment, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<int *(int), access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<int *(int), alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<int &, access::address_space::private_space, (decorated)0>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<int &, alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'sycl::multi_ptr<float, sycl::access::address_space::local_space, (decorated)0>'}}
  aligned_private_alloca_bad_6<float, alignment, size, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align can only return 'sycl::private_ptr' to a cv-unqualified trivial type. Got 'multi_ptr<non_trivial, access::address_space::private_space, (decorated)1>'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<non_trivial, alignment, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align must be passed a specialization constant of integral value type as a template argument. Got 'int'}}
  aligned_private_alloca_bad_7<float, alignment, int, sycl::access::decorated::no>(h);

  // expected-error@+1 {{__builtin_intel_sycl_alloca_with_align must be passed a specialization constant of integral value type as a template argument. Got 'const sycl::specialization_id<float> &'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, alignment, badsize, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{requested alignment is not a power of 2}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, 3, size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{requested alignment must be 268435455 or smaller}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, exp2<size_t>(60), size, sycl::access::decorated::yes>(h);

  // expected-error@+1 {{requested alignment is less than minimum alignment of 4 for type 'float'}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, 1, size, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca_with_align expects a specialization constant with a default value of at least one as an argument. Got 0}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, alignof(float), zero, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca_with_align expects a specialization constant with a default value of at least one as an argument. Got -1}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, alignof(float), negative, sycl::access::decorated::yes>(h);

  // expected-warning@+1 {{__builtin_intel_sycl_alloca_with_align expects a specialization constant with a default value of at least one as an argument. Got -9}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<float, alignof(float), negative_expr_ref, sycl::access::decorated::yes>(h);
}
