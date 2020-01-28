// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

void defined() {
  // empty
}

void undefined();

SYCL_EXTERNAL void undefinedExternal();

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class CallToUndefinedFnTester>([]() {
    defined();
    undefinedExternal();
    undefined();
    // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
  });
}
