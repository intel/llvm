// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only %s
// expected-no-diagnostics

void F(__float128);

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  __float128 A;
  kernel_single_task<class fake_kernel>([=]() {});
  return 0;
}
