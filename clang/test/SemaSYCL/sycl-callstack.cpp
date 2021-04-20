// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -verify -Wno-sycl-2017-compat -fsyntax-only -std=c++17 %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-note@+1 {{called by 'kernel_single_task}}
  kernelFunc();
}

void foo() {
    // expected-error@+1 {{SYCL kernel cannot use exceptions}}
   throw 3;
}

int main() {
// expected-note@+1 {{called by 'operator()'}}
kernel_single_task<class fake_kernel>([]() { foo(); });
}
