// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -fsycl-is-host -verify %s -DSYCLHOST
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

#ifdef SYCLHOST
// expected-no-diagnostics
#endif

void foo()
{
  #ifndef SYCLHOST
  // expected-warning@+2 {{'doublepump' attribute ignored}}
  #endif
  [[intelfpga::doublepump]] unsigned int v_one[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'memory' attribute ignored}}
  #endif
  [[intelfpga::memory]] unsigned int v_two[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'register' attribute ignored}}
  #endif
  [[intelfpga::register]] unsigned int v_three[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'singlepump' attribute ignored}}
  #endif
  [[intelfpga::singlepump]] unsigned int v_four[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'bankwidth' attribute ignored}}
  #endif
  [[intelfpga::bankwidth(4)]] unsigned int v_five[32];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'numbanks' attribute ignored}}
  #endif
  [[intelfpga::numbanks(8)]] unsigned int v_six[32];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'max_private_copies' attribute ignored}}
  #endif
  [[intelfpga::max_private_copies(8)]] unsigned int v_seven[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'merge' attribute ignored}}
  #endif
  [[intelfpga::merge("mrg1","depth")]]  unsigned int v_eight[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'max_replicates' attribute ignored}}
  #endif
  [[intelfpga::max_replicates(2)]]
  unsigned int v_nine[64];

  #ifndef SYCLHOST
  // expected-warning@+2 {{'simple_dual_port' attribute ignored}}
  #endif
  [[intelfpga::simple_dual_port]]
  unsigned int v_ten[64];
}

