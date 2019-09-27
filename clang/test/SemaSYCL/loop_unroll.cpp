// RUN: %clang_cc1 -x c++ -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s

void foo() {
  // expected-error@+1 {{clang loop attributes must be applied to for, while, or do statements}}
  [[clang::loop_unroll(8)]] int a[10];

  // expected-error@+1 {{'loop_unroll' attribute takes no more than 1 argument}}
  [[clang::loop_unroll(2,2)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+1 {{'loop_unroll' attribute requires a positive integral compile time constant expression}}
  [[clang::loop_unroll(0)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+1 {{'loop_unroll' attribute requires a positive integral compile time constant expression}}
  [[clang::loop_unroll(-2)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+1 {{'loop_unroll' attribute requires an integer constant}}
  [[clang::loop_unroll("str123")]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+1 {{duplicate unroll loop attribute 'loop_unroll'}}
  [[clang::loop_unroll(2)]]
  [[clang::loop_unroll(4)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+2 {{incompatible loop unroll instructions: '#pragma unroll(4)' and '[[clang::loop_unroll(2)]]'}}
#pragma unroll 4
  [[clang::loop_unroll(2)]]
  for (int i = 0; i < 10; ++i);

  // no error expected
  [[clang::loop_unroll(4)]]
  [[intelfpga::ii(2)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+2 {{'loop_unroll' attribute requires an integer constant}}
  int b = 4;
  [[clang::loop_unroll(b)]]
  for (int i = 0; i < 10; ++i);

  // no error expected
  constexpr int c = 4;
  [[clang::loop_unroll(c)]]
  for (int i = 0; i < 10; ++i);
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
  });
  return 0;
}
