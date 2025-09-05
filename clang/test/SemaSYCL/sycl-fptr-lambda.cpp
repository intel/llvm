// RUN: %clang_cc1 -fsycl-is-device -std=c++14 -verify -fsyntax-only %s
// expected-no-diagnostics

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {
    auto l = [](auto f) { f(); };
  });
  return 0;
}
