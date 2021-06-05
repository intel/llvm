// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-linux-gnu -fsycl-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -fsyntax-only %s
// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-linux-gnu -fsycl-is-device -fsyntax-only -mlong-double-64 %s

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  // expected-note@+1 {{called by 'kernel<variables}}
  kernelFunc();
}

//expected-note@+1 {{'foo' defined here}}
void foo(long double A) {}

int main() {
  //expected-note@+1 {{'CapturedToDevice' defined here}}
  long double CapturedToDevice = 1;
  kernel<class variables>([=]() {
    // expected-error@+2 {{'foo' requires 128 bit size 'long double' type support, but device 'spir64' does not support it}}
    // expected-error@+1 {{'CapturedToDevice' requires 128 bit size 'long double' type support, but device 'spir64' does not support it}}
    foo(CapturedToDevice);
  });

  return 0;
}
