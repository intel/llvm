// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -Wno-return-type -Wno-sycl-2017-compat -Wno-error=sycl-strict -verify -fsyntax-only -std=c++17 %s

// This recursive function is not called from sycl kernel,
// so it should not be diagnosed.
int fib(int n) {
  if (n <= 1)
    return n;
  return fib(n - 1) + fib(n - 2);
}

// expected-note@+1 2{{function implemented using recursion declared here}}
void kernel2(void) {
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  kernel2();
}

using myFuncDef = int(int, int);

typedef __typeof__(sizeof(int)) size_t;

SYCL_EXTERNAL
void *operator new(size_t);

void usage2(myFuncDef functionPtr) {
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ip = new int;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  kernel2();
}

int addInt(int n, int m) {
  return n + m;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-note@+1 {{called by 'kernel_single_task}}
  kernelFunc();
}

int main() {
  // expected-note@+1 {{called by 'operator()'}}
  kernel_single_task<class fake_kernel>([]() { usage2(&addInt); });
  return fib(5);
}
