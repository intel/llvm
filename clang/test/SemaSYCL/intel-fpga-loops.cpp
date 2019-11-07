// RUN: %clang_cc1 -x c++ -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s

// Test for Intel FPGA loop attributes applied not to a loop
void foo() {
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::ivdep]] int a[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::ivdep(2)]] int b[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::ii(2)]] int c[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::max_concurrency(2)]] int d[10];

  int arr[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::ivdep(arr)]] int e[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while, or do statements}}
  [[intelfpga::ivdep(arr, 2)]] int f[10];
}

// Test for incorrect number of arguments for Intel FPGA loop attributes
void boo() {
  int a[10];
  int b[10];
  // expected-error@+1 {{duplicate argument to 'ivdep'. attribute requires one or both of a safelen and array}}
  [[intelfpga::ivdep(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'ii' attribute takes at least 1 argument - attribute ignored}}
  [[intelfpga::ii]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'ii' attribute takes no more than 1 argument - attribute ignored}}
  [[intelfpga::ii(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'max_concurrency' attribute takes at least 1 argument - attribute ignored}}
  [[intelfpga::max_concurrency]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'max_concurrency' attribute takes no more than 1 argument - attribute ignored}}
  [[intelfpga::max_concurrency(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // expected-error@+1 {{duplicate argument to 'ivdep'. attribute requires one or both of a safelen and array}}
  [[intelfpga::ivdep(2, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{duplicate argument to 'ivdep'. attribute requires one or both of a safelen and array}}
  [[intelfpga::ivdep(a, b)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'. Expected integer or array variable}}
  [[intelfpga::ivdep(2, 3.0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect argument value for Intel FPGA loop attributes
void goo() {
  int a[10];
  // no diagnostics are expected
  [[intelfpga::max_concurrency(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ivdep' attribute requires a positive integral compile time constant expression}}
  [[intelfpga::ivdep(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'ii' attribute requires a positive integral compile time constant expression - attribute ignored}}
  [[intelfpga::ii(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-warning@+1 {{'max_concurrency' attribute requires a non-negative integral compile time constant expression - attribute ignored}}
  [[intelfpga::max_concurrency(-1)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'. Expected integer or array variable}}
  [[intelfpga::ivdep("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ii' attribute requires an integer constant}}
  [[intelfpga::ii("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute requires an integer constant}}
  [[intelfpga::max_concurrency("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'. Expected integer or array variable}}
  [[intelfpga::ivdep("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intelfpga::ivdep(a, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intelfpga::ivdep(2, a)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for Intel FPGA loop attributes duplication
void zoo() {
  int a[10];
  // no diagnostics are expected
  [[intelfpga::ivdep]]
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  [[intelfpga::ivdep(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep(4)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::max_concurrency(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'max_concurrency'}}
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ii(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ii'}}
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ii(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ii'}}
  [[intelfpga::max_concurrency(2)]]
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  [[intelfpga::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep(a)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep(4)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // no diagnostics are expected
  [[intelfpga::ivdep(a)]]
  [[intelfpga::ivdep(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  [[intelfpga::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // Ensure we only diagnose conflict with the 'worst', not all.
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  [[intelfpga::ivdep(3)]]
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 4}}
  [[intelfpga::ivdep(4)]]
  // expected-note@+1 2 {{previous attribute is here}}
  [[intelfpga::ivdep(5)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  [[intelfpga::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 3 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep(a, 3)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template<int A, int B, int C>
void ivdep_dependent() {
  int a[10];
  // test this again to ensure we skip properly during instantiation.
  [[intelfpga::ivdep(3)]]
  // expected-warning@-1 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intelfpga::ivdep(5)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  [[intelfpga::ivdep(C)]]
  // expected-error@-1 {{'ivdep' attribute requires a positive integral compile time constant expression}}
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // expected-warning@+3 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intelfpga::ivdep(A)]]
  [[intelfpga::ivdep(B)]]
  // expected-warning@-2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  (void)[]() {
  // expected-warning@+3 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intelfpga::ivdep]]
  [[intelfpga::ivdep]]
    while(true);
  };
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
    boo();
    goo();
    zoo();
    ivdep_dependent<4, 2, 1>();
    //expected-note@-1 +{{in instantiation of function template specialization}}
    ivdep_dependent<2, 4, -1>();
    //expected-note@-1 +{{in instantiation of function template specialization}}
  });
  return 0;
}
