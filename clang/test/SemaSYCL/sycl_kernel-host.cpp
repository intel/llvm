// RUN: %clang_cc1 -fsycl -fsycl-is-host -fsyntax-only -verify %s

// expected-warning@+1 {{'sycl_kernel' attribute only applies to function templates}}
void __attribute__((sycl_kernel))
KernelImpl1() {
}

// expected-warning@+3 {{function template with 'sycl_kernel' attribute must have at least one parameter}}
template <typename Func>
void __attribute__((sycl_kernel))
KernelImpl2() {
}

// expected-warning@+3 {{function template with 'sycl_kernel' attribute must have a 'void' return type}}
template <typename Func>
int __attribute__((sycl_kernel))
KernelImpl3(Func f) {
  f();
}

// expected-no-error
template <typename Func>
void __attribute__((sycl_kernel))
KernelImpl4(Func f, int i, double d) {
  f(i, d);
}

template <typename Name, typename Func>
void __attribute__((sycl_kernel))
Kernel(Func f) {
  KernelImpl4(f, 1, 2.0);
}

void func() {
  auto Lambda = [](int i, double d) { d += i; };
  Kernel<class Foo>(Lambda);
}
