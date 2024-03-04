// RUN: %clang_cc1 -fsyntax-only -fsycl-is-host -triple x86_64-unknown-unknown -verify -Wpedantic -fcxx-exceptions %s

#include <stddef.h>

#include "Inputs/sycl.hpp"

constexpr sycl::specialization_id<size_t> size(1);

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
sycl::multi_ptr<ElementType, sycl::access::address_space::private_space, DecorateAddress>
private_alloca_bad_0(sycl::kernel_handler &h);

void test(sycl::kernel_handler &h) {
  // expected-error@+1 {{'__builtin_intel_sycl_alloca' is only available in SYCL device}}
  private_alloca_bad_0<float, size, sycl::access::decorated::no>(h);
}
