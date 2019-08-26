// RUN: %clang_cc1 -verify -fsyntax-only -fsycl-is-device %s

extern __thread void* __once_callable;  // expected-no-error
extern __thread void (*__once_call)();  // expected-no-error

void usage() {
  // expected-error@+2{{thread-local storage is not supported for the current target}}
  // expected-error@+1{{SYCL kernel cannot use a global variable}}
  __once_callable = 0;
  // expected-error@+3{{thread-local storage is not supported for the current target}}
  // expected-error@+2{{SYCL kernel cannot use a global variable}}
  // expected-error@+1{{SYCL kernel cannot call through a function pointer}}
  __once_call();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}
