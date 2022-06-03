// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -verify -pedantic %s

// The test checks functionality of [[intel::reqd_sub_group_size()]] attribute on SYCL kernel.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::reqd_sub_group_size(4)]] void foo() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(32)]] void baz() {} // expected-note {{conflicting attribute is here}}

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[intel::reqd_sub_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    foo();
  }
};

int main() {
  q.submit([&](handler &h) {
    Functor8 f8;
    h.single_task<class kernel_name1>(f8);

    h.single_task<class kernel_name2>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      foo();
      baz();
    });
  });
  return 0;
}
[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B();
[[intel::reqd_sub_group_size(16)]] void A() {
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B() {
  A();
}

// expected-note@+1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(2)]] void sg_size2() {}

// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{conflicting attributes applied to a SYCL kernel}}
[[intel::reqd_sub_group_size(4)]] __attribute__((sycl_device)) void sg_size4() {
  sg_size2();
}
