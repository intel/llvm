// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -sycl-std=2017 -verify %s 

// Test that we get ignored attribute warning when using
// a [[intel::named_sub_group_size()]] attribute spelling while not
// in SYCL 2020 mode.
[[intel::named_sub_group_size(automatic)]] void func_ignore(); // expected-warning {{'named_sub_group_size' attribute ignored}}
