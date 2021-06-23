// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -sycl-std=2017 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -verify=sycl-2017 %s

// We do not expect any diagnostics from this file when disabling future compat
// warnings.
// sycl-2017-no-diagnostics

// Test that we get compatibility warnings when using a SYCL 2020 attribute
// spelling while not in SYCL 2020 mode.
[[sycl::reqd_work_group_size(1, 1, 1)]] void f1(); // expected-warning {{use of attribute 'reqd_work_group_size' is a SYCL 2020 extension}}
[[sycl::work_group_size_hint(1, 1, 1)]] void f2(); // expected-warning {{use of attribute 'work_group_size_hint' is a SYCL 2020 extension}}
[[sycl::reqd_sub_group_size(1)]] void f3(); // expected-warning {{use of attribute 'reqd_sub_group_size' is a SYCL 2020 extension}}
[[sycl::vec_type_hint(int)]] void f4(); // expected-warning {{use of attribute 'vec_type_hint' is a SYCL 2020 extension}}
