// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -triple spir64_x86_64 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -triple spir64_gen -verify %s

// Test clang emits correct errors on 'private_alloca' used on an AOT SPIR target.

#include <stddef.h>

#include "Inputs/sycl.hpp"
#include "Inputs/private_alloca.hpp"

struct myStruct {
  int a;
  int b;
};

constexpr sycl::specialization_id<size_t> size(1);
constexpr sycl::specialization_id<int> intSize(1);
constexpr sycl::specialization_id<unsigned short> shortSize(1);

void basic_test(sycl::kernel_handler &kh) {
  // expected-error@+2 {{builtin is not supported on this target}}
  // expected-note@+1 {{__builtin_intel_sycl_alloca cannot be AOT compiled}}
  sycl::ext::oneapi::experimental::private_alloca<
    int, size, sycl::access::decorated::yes>(kh);
  // expected-error@+2 {{builtin is not supported on this target}}
  // expected-note@+1 {{__builtin_intel_sycl_alloca cannot be AOT compiled}}
  sycl::ext::oneapi::experimental::private_alloca<
    float, intSize, sycl::access::decorated::no>(kh);
  // expected-error@+2 {{builtin is not supported on this target}}
  // expected-note@+1 {{__builtin_intel_sycl_alloca cannot be AOT compiled}}
  sycl::ext::oneapi::experimental::private_alloca<
    myStruct, shortSize, sycl::access::decorated::legacy>(kh);
  // expected-error@+2 {{builtin is not supported on this target}}
  // expected-note@+1 {{__builtin_intel_sycl_alloca_with_align cannot be AOT compiled}}
  sycl::ext::oneapi::experimental::aligned_private_alloca<
    myStruct, alignof(myStruct) * 2, shortSize, sycl::access::decorated::legacy>(kh);
}
