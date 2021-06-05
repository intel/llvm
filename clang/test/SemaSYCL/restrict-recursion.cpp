// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -Wno-return-type -verify -Wno-sycl-2017-compat -fsyntax-only -std=c++17 %s

// This recursive function is not called from sycl kernel,
// so it should not be diagnosed.
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}

typedef struct S {
template <typename T>
  // expected-note@+1 3{{function implemented using recursion declared here}}
T factT(T i, T j)
{
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  return factT(j,i);
}

int fact(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return factT<unsigned>(i-1, i);
}
} S_type;


  // expected-note@+1 2{{function implemented using recursion declared here}}
int fact(unsigned i);
  // expected-note@+1 2{{function implemented using recursion declared here}}
int fact1(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return fact(i-1) * i;
}
int fact(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return fact1(i-1) * i;
}

bool isa_B(void) {
  S_type s;

  unsigned f = s.fact(3);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned f1 = s.factT<unsigned>(3,4);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned g = fact(3);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned g1 = fact1(3);
  return 0;
}

template <typename N, typename L>
__attribute__((sycl_kernel)) void kernel(const L &l) {
  l();
}

// expected-note@+1 3{{function implemented using recursion declared here}}
void kernel2_recursive(void) {
  // expected-error@+1 1{{SYCL kernel cannot call a recursive function}}
  kernel2_recursive();
}

using myFuncDef = int(int,int);

void usage(myFuncDef functionPtr) {
  kernel<class kernel1>([]() { isa_B(); });
}
void usage2(  myFuncDef functionPtr ) {
  // expected-error@+1 2{{SYCL kernel cannot call a recursive function}}
  kernel<class kernel2>([]() { kernel2_recursive(); });
}
void usage3(  myFuncDef functionPtr ) {
  kernel<class kernel3>([]() { ; });
}

int addInt(int n, int m) {
    return n+m;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

template <typename name, typename Func>
// expected-note@+1 2{{function implemented using recursion declared here}}
__attribute__((sycl_kernel)) void kernel_single_task2(const Func &kernelFunc) {
  kernelFunc();
  // expected-error@+1 2{{SYCL kernel cannot call a recursive function}}
  kernel_single_task2<name, Func>(kernelFunc);
}

int main() {
  kernel_single_task<class fake_kernel>([]() { usage(  &addInt ); });
  kernel_single_task<class fake_kernel>([]() { usage2(  &addInt ); });
  kernel_single_task2<class fake_kernel>([]() { usage3(  &addInt ); });
  return fib(5);
}

