// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s -DSYCLHOST
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

#ifdef SYCLHOST
// expected-no-diagnostics
#endif

void foo()
{
#ifndef SYCLHOST
// expected-warning@+2 {{'doublepump' attribute ignored}}
#endif
  [[intel::doublepump]] unsigned int v_one[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'fpga_memory' attribute ignored}}
#endif
  [[intel::fpga_memory]] unsigned int v_two[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'fpga_register' attribute ignored}}
#endif
  [[intel::fpga_register]] unsigned int v_three[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'singlepump' attribute ignored}}
#endif
  [[intel::singlepump]] unsigned int v_four[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'bankwidth' attribute ignored}}
#endif
  [[intel::bankwidth(4)]] unsigned int v_five[32];

#ifndef SYCLHOST
// expected-warning@+2 {{'numbanks' attribute ignored}}
#endif
  [[intel::numbanks(8)]] unsigned int v_six[32];

#ifndef SYCLHOST
// expected-warning@+2 {{'private_copies' attribute ignored}}
#endif
  [[intel::private_copies(8)]] unsigned int v_seven[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'merge' attribute ignored}}
#endif
  [[intel::merge("mrg1", "depth")]] unsigned int v_eight[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'max_replicates' attribute ignored}}
#endif
  [[intel::max_replicates(2)]] unsigned int v_nine[64];

#ifndef SYCLHOST
// expected-warning@+2 {{'simple_dual_port' attribute ignored}}
#endif
  [[intel::simple_dual_port]] unsigned int v_ten[64];
}

