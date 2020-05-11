// RUN: %clang_cc1 -fsycl -fsycl-is-host -fsyntax-only -verify %s -DSYCLHOST
// RUN: %clang_cc1 -fsyntax-only -verify %s

#ifdef SYCLHOST
// expected-no-diagnostics
#endif

void foo()
{
  #ifndef SYCLHOST
  // expected-warning@+2 {{'intel_reqd_sub_group_size' attribute ignored}}
  #endif
  [[cl::intel_reqd_sub_group_size(4)]] void m();
}
