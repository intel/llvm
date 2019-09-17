// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -fsyntax-only -verify %s

void bar() { throw 5; } // expected-no-error
void foo() { throw 10; } // expected-error {{SYCL kernel cannot use exceptions}}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // expected-note@+1 {{called by 'operator()'}}
  kernel_single_task<class fake_kernel>([]() { foo(); });
  bar();
}
