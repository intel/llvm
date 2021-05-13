// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify -pedantic %s

#include "sycl.hpp"

sycl::queue deviceQueue;

// Test for Intel FPGA loop attributes applied not to a loop
void foo() {
  // expected-error@+1 {{'ivdep' attribute cannot be applied to a declaration}}
  [[intel::ivdep]] int a[10];
  // expected-error@+1 {{'initiation_interval' attribute only applies to 'for', 'while', 'do' statements, and functions}}
  [[intel::initiation_interval(2)]] int c[10];
  // expected-error@+1 {{'max_concurrency' attribute only applies to 'for', 'while', 'do' statements, and functions}}
  [[intel::max_concurrency(2)]] int d[10];
  // expected-error@+1 {{'disable_loop_pipelining' attribute only applies to 'for', 'while', 'do' statements, and functions}}
  [[intel::disable_loop_pipelining]] int g[10];
  // expected-error@+1 {{'loop_coalesce' attribute cannot be applied to a declaration}}
  [[intel::loop_coalesce(2)]] int h[10];
  // expected-error@+1 {{'max_interleaving' attribute cannot be applied to a declaration}}
  [[intel::max_interleaving(4)]] int i[10];
  // expected-error@+1 {{'speculated_iterations' attribute cannot be applied to a declaration}}
  [[intel::speculated_iterations(6)]] int j[10];
  // expected-error@+1 {{'nofusion' attribute cannot be applied to a declaration}}
  [[intel::nofusion]] int k[10];
  // expected-error@+1{{'loop_count_avg' attribute cannot be applied to a declaration}}
  [[intel::loop_count_avg(6)]] int p[10];
}

// Test for deprecated spelling of Intel FPGA loop attributes
void foo_deprecated() {
  int a[10];
  // expected-warning@+2 {{attribute 'intelfpga::ivdep' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::ivdep' instead?}}
  [[intelfpga::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::ii' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::initiation_interval' instead?}}
  [[intelfpga::ii(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intel::ii' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::initiation_interval' instead?}}
  [[intel::ii(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::max_concurrency' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::max_concurrency' instead?}}
  [[intelfpga::max_concurrency(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::max_interleaving' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::max_interleaving' instead?}}
  [[intelfpga::max_interleaving(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::disable_loop_pipelining' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::disable_loop_pipelining' instead?}}
  [[intelfpga::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::loop_coalesce' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::loop_coalesce' instead?}}
  [[intelfpga::loop_coalesce(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+2 {{attribute 'intelfpga::speculated_iterations' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::speculated_iterations' instead?}}
  [[intelfpga::speculated_iterations(6)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect number of arguments for Intel FPGA loop attributes
void boo() {
  int a[10];
  int b[10];
  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(2, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'initiation_interval' attribute takes one argument}}
  [[intel::initiation_interval]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'initiation_interval' attribute takes one argument}}
  [[intel::initiation_interval(2, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute takes one argument}}
  [[intel::max_concurrency]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute takes one argument}}
  [[intel::max_concurrency(2, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(2, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(a, b)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep(2, 3.0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{'disable_loop_pipelining' attribute takes no arguments}}
  [[intel::disable_loop_pipelining(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_coalesce' attribute takes no more than 1 argument}}
  [[intel::loop_coalesce(2, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_interleaving' attribute takes one argument}}
  [[intel::max_interleaving]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_interleaving' attribute takes one argument}}
  [[intel::max_interleaving(2, 4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'speculated_iterations' attribute takes one argument}}
  [[intel::speculated_iterations]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'speculated_iterations' attribute takes one argument}}
  [[intel::speculated_iterations(1, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'nofusion' attribute takes no arguments}}
  [[intel::nofusion(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_count_avg' attribute takes one argument}}
  [[intel::loop_count_avg(3, 6)]]  for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect argument value for Intel FPGA loop attributes
void goo() {
  int a[10];
  // no diagnostics are expected
  [[intel::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::max_concurrency(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'ivdep' attribute requires a positive integral compile time constant expression}}
  [[intel::ivdep(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'initiation_interval' attribute requires a positive integral compile time constant expression}}
  [[intel::initiation_interval(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute requires a non-negative integral compile time constant expression}}
  [[intel::max_concurrency(-1)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_coalesce' attribute requires a positive integral compile time constant expression}}
  [[intel::loop_coalesce(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_interleaving' attribute requires a non-negative integral compile time constant expression}}
  [[intel::max_interleaving(-1)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'speculated_iterations' attribute requires a non-negative integral compile time constant expression}}
  [[intel::speculated_iterations(-1)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'initiation_interval' attribute requires an integer constant}}
  [[intel::initiation_interval("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute requires an integer constant}}
  [[intel::max_concurrency("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_coalesce' attribute requires an integer constant}}
  [[intel::loop_coalesce("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'max_interleaving' attribute requires an integer constant}}
  [[intel::max_interleaving("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'speculated_iterations' attribute requires an integer constant}}
  [[intel::speculated_iterations("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(a, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(2, a)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  int *ptr;
  // no diagnostics are expected
  [[intel::ivdep(2, ptr)]] for (int i = 0; i != 10; ++i)
      ptr[i] = 0;

  struct S {
    int arr[10];
    int *ptr;
  } s;

  // no diagnostics are expected
  [[intel::ivdep(2, s.arr)]] for (int i = 0; i != 10; ++i)
      s.arr[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(2, s.ptr)]] for (int i = 0; i != 10; ++i)
      s.ptr[i] = 0;

  // no diagnostics are expected
  [[intel::nofusion]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::loop_count_avg(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_count_avg' attribute requires a non-negative integral compile time constant expression}}
  [[intel::loop_count_avg(-1)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'loop_count_avg' attribute requires an integer constant}}
    [[intel::loop_count_avg("abc")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for Intel FPGA loop attributes duplication
void zoo() {
  int a[10];
  // no diagnostics are expected
  [[intel::ivdep]]
  [[intel::max_concurrency(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::max_concurrency(2)]]
  // expected-error@+1 {{duplicate Intel FPGA loop attribute 'max_concurrency'}}
  [[intel::max_concurrency(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::initiation_interval(2)]]
  // expected-error@+1 {{duplicate Intel FPGA loop attribute 'initiation_interval'}}
  [[intel::initiation_interval(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::initiation_interval(2)]]
  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'initiation_interval'}}
  [[intel::max_concurrency(2)]]
  [[intel::initiation_interval(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::disable_loop_pipelining]]
  // expected-error@+1 {{duplicate Intel FPGA loop attribute 'disable_loop_pipelining'}}
  [[intel::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::loop_coalesce(2)]]
  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'loop_coalesce'}}
  [[intel::max_interleaving(1)]]
  [[intel::loop_coalesce]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::max_interleaving(1)]]
  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'max_interleaving'}}
  [[intel::speculated_iterations(1)]]
  [[intel::max_interleaving(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::speculated_iterations(1)]]
  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'speculated_iterations'}}
  [[intel::loop_coalesce]]
  [[intel::speculated_iterations(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(a)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // no diagnostics are expected
  [[intel::ivdep(a)]]
  [[intel::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // Ensure we only diagnose conflict with the 'worst', not all.
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  [[intel::ivdep(3)]]
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 4}}
  [[intel::ivdep(4)]]
  // expected-note@+1 2 {{previous attribute is here}}
  [[intel::ivdep(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 3 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(a, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::nofusion]]
  // expected-error@+1 {{duplicate Intel FPGA loop attribute 'nofusion'}}
  [[intel::nofusion]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::loop_count_avg(2)]]
  // expected-error@+1{{duplicate Intel FPGA loop attribute 'loop_count_avg'}}
  [[intel::loop_count_avg(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for Intel FPGA loop attributes compatibility
void loop_attrs_compatibility() {
  int a[10];
  // no diagnostics are expected
  [[intel::disable_loop_pipelining]]
  [[intel::loop_coalesce]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+3 {{'max_interleaving' and 'disable_loop_pipelining' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[intel::disable_loop_pipelining]]
  [[intel::max_interleaving(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+3 {{'disable_loop_pipelining' and 'speculated_iterations' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[intel::speculated_iterations(0)]]
  [[intel::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+3 {{'max_concurrency' and 'disable_loop_pipelining' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[intel::disable_loop_pipelining]]
  [[intel::max_concurrency(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+3 {{'disable_loop_pipelining' and 'initiation_interval' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[intel::initiation_interval(10)]]
  [[intel::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+3 {{'ivdep' and 'disable_loop_pipelining' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[intel::disable_loop_pipelining]]
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::disable_loop_pipelining]]
  [[intel::nofusion]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::disable_loop_pipelining]]
  [[intel::loop_count_avg(8)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::loop_count_min(8)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::loop_count_max(8)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template<int A, int B, int C>
void ivdep_dependent() {
  int a[10];
  // test this again to ensure we skip properly during instantiation.
  [[intel::ivdep(3)]]
  // expected-warning@-1 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intel::ivdep(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep(C)]]
  // expected-error@-1 {{'ivdep' attribute requires a positive integral compile time constant expression}}
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+3 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(A)]]
  [[intel::ivdep(B)]]
  // expected-warning@-2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  (void)[]() {
  // expected-warning@+3 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intel::ivdep]]
  [[intel::ivdep]] while (true);
  };
}

template <int A, int B, int C>
void ii_dependent() {
  int a[10];
  // expected-error@+1 {{'initiation_interval' attribute requires a positive integral compile time constant expression}}
  [[intel::initiation_interval(C)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'initiation_interval'}}
  [[intel::initiation_interval(A)]]
  [[intel::initiation_interval(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A, int B, int C>
void max_concurrency_dependent() {
  int a[10];
  // expected-error@+1 {{'max_concurrency' attribute requires a non-negative integral compile time constant expression}}
  [[intel::max_concurrency(C)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'max_concurrency'}}
  [[intel::max_concurrency(A)]]
  [[intel::max_concurrency(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template<int A, int B, int C>
void loop_count_control_dependent() {
  int a[10];

  //expected-error@+1{{'loop_count_avg' attribute requires a non-negative integral compile time constant expression}}
  [[intel::loop_count_avg(C)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  //expected-error@+1{{'loop_count_min' attribute requires a non-negative integral compile time constant expression}}
  [[intel::loop_count_min(C)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  //expected-error@+1{{'loop_count_max' attribute requires a non-negative integral compile time constant expression}}
  [[intel::loop_count_max(C)]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::loop_count_avg(A)]]
  //expected-error@+1{{duplicate Intel FPGA loop attribute 'loop_count_avg'}}
  [[intel::loop_count_avg(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::loop_count_min(A)]]
  //expected-error@+1{{duplicate Intel FPGA loop attribute 'loop_count_min'}}
  [[intel::loop_count_min(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::loop_count_max(A)]]
  //expected-error@+1{{duplicate Intel FPGA loop attribute 'loop_count_max'}}
  [[intel::loop_count_max(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

}

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      foo();
      foo_deprecated();
      boo();
      goo();
      zoo();
      loop_attrs_compatibility();
      ivdep_dependent<4, 2, 1>();
      //expected-note@-1 +{{in instantiation of function template specialization}}
      ivdep_dependent<2, 4, -1>();
      //expected-note@-1 +{{in instantiation of function template specialization}}
      ii_dependent<2, 4, -1>();
      //expected-note@-1 +{{in instantiation of function template specialization}}
      max_concurrency_dependent<1, 4, -2>();
      //expected-note@-1 +{{in instantiation of function template specialization}}

     loop_count_control_dependent<3, 2, -1>();
      //expected-note@-1{{in instantiation of function template specialization 'loop_count_control_dependent<3, 2, -1>' requested here}}
});
  });

  return 0;
}

void parse_order_error() {
  // We had a bug where we would only look at the first attribute in the group
  // when trying to determine whether to diagnose the loop attributes on an
  // incorrect subject. Test that we properly catch this situation.
  [[clang::nomerge, intel::max_concurrency(1)]] // expected-error {{'max_concurrency' attribute only applies to 'for', 'while', 'do' statements, and functions}}
  if (1) { parse_order_error(); }               // Recursive call silences unrelated diagnostic about nomerge.

  [[clang::nomerge, intel::max_concurrency(1)]] // OK
  while (1) { parse_order_error(); }            // Recursive call silences unrelated diagnostic about nomerge.
}
