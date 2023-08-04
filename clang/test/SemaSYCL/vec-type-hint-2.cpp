// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -internal-isystem %S/Inputs -fsyntax-only -verify %s

// Test which verifies [[sycl::vec_type_hint()]] is accepted
// with non-conforming lambda syntax.

// NOTE: This attribute is not supported in the SYCL backends.
// To be minimally conformant with SYCL2020, attribute is
// accepted by the Clang FE with a warning. No additional
// semantic handling or IR generation is done for this
// attribute.

#include "sycl.hpp"

struct test {};

using namespace sycl;
queue q;

void bar() {
  q.submit([&](handler &h) {
    h.single_task<class kernelname>(
        // expected-warning@+1 {{attribute 'vec_type_hint' is deprecated; attribute ignored}}
        []() [[sycl::vec_type_hint(test)]] {});
  });
}

