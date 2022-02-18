// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

[[intel::reqd_sub_group_size(8)]] void fun() {}

class Functor {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() {}
};

// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s -DSYCLHOST
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

#ifdef SYCLHOST
// expected-no-diagnostics
#endif

void foo()
{
#ifndef SYCLHOST
// expected-warning@+2 {{'reqd_sub_group_size' attribute ignored}}
#endif
  class Functor {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() {}
};
}

