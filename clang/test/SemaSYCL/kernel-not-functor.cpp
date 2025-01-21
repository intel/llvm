// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// Disabled while we are no longer checking in host mode.
// RUNX: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

template <typename Name, typename F>
__attribute__((sycl_kernel)) void kernel(F kernelFunc) {
  kernelFunc();
}

template <typename Name, typename F>
void uses_kernel(F kernelFunc) {
  // expected-error@+1{{kernel parameter must be a lambda or function object}}
  kernel<Name>(kernelFunc);
}

void func() {}

template <typename Name>
void kernel_wrapper() {
  // expected-note@+1{{in instantiation of function template specialization}}
  uses_kernel<Name>(func);
}

void use() {
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_wrapper<class Foo>();
}
