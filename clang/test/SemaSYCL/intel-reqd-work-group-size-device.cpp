// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -fsyntax-only -verify %s

// The test checks support and functionality of reqd_work_group_size kernel attribute in SYCL 2017.

#include "sycl.hpp"

using namespace sycl;
queue q;

[[sycl::reqd_work_group_size(4)]] void f4() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32)]] void f32() {} // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16)]] void f16() {}      // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16, 16)]] void f16x16() {} // expected-note {{conflicting attribute is here}}

[[sycl::reqd_work_group_size(32, 32)]] void f32x32() {}      // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}

[[intel::reqd_work_group_size(4, 2, 9)]] void unknown() {} // expected-warning{{unknown attribute 'reqd_work_group_size' ignored}}

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[sycl::reqd_work_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    f4();
  }
};

// Tests of redeclaration of [[intel::max_work_group_size()]] and [[sycl::reqd_work_group_size()]] - expect error
[[intel::max_work_group_size(4, 4, 4)]] void func2();   // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(8, 8, 8)]] void func2() {} // expected-error {{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}

[[sycl::reqd_work_group_size(4, 4, 4)]] void func3();   // expected-note {{previous attribute is here}}
[[sycl::reqd_work_group_size(1, 1, 1)]] void func3() {} // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}

int main() {
  q.submit([&](handler &h) {
    Functor8 f8;
    h.single_task<class kernel_name1>(f8);

    h.single_task<class kernel_name2>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f4();
      f32();
    });

    h.single_task<class kernel_name3>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f16();
      f16x16();
    });

    h.single_task<class kernel_name4>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f32x32x32();
      f32x32();
    });

    // expected-error@+1 {{expected variable name or 'this' in lambda capture list}}
    h.single_task<class kernel_name5>([[sycl::reqd_work_group_size(32, 32, 32)]][]() {
      f32x32x32();
    });

    h.single_task<class kernel_name6>(
        []() { func2(); });

    h.single_task<class kernel_name7>(
        []() { func3(); });
  });
  return 0;
}
