// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -triple spir64-unknown-unknown -verify -Wpedantic                  %s

// Test verification of __builtin_intel_sycl_alloca when used in different valid ways.

#include <stddef.h>

#include "Inputs/sycl.hpp"
#include "Inputs/private_alloca.hpp"

// expected-no-diagnostics

struct myStruct {
  int a;
  int b;
};

constexpr sycl::specialization_id<size_t> size(1);
constexpr sycl::specialization_id<int> intSize(-1);
constexpr sycl::specialization_id<unsigned short> shortSize(1);

void basic_test(sycl::kernel_handler &kh) {
  sycl::ext::oneapi::experimental::private_alloca<
    int, size, sycl::access::decorated::yes>(kh);
  sycl::ext::oneapi::experimental::private_alloca<
    float, intSize, sycl::access::decorated::no>(kh);
  sycl::ext::oneapi::experimental::private_alloca<
    myStruct, shortSize, sycl::access::decorated::legacy>(kh);
}
