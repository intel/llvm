// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

template <int A>
void bar() {
  // expected-error@+1 {{'loop_unroll' attribute requires a positive integral compile time constant expression}}
  [[clang::loop_unroll(A)]]
  for (int i = 0; i < 10; ++i);
}

void foo() {
  // expected-error@+1 {{'loop_unroll' attribute cannot be applied to a declaration}}
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

  // expected-error@+2 {{duplicate unroll loop attribute 'loop_unroll'}}
  [[clang::loop_unroll(2)]]
  [[clang::loop_unroll(4)]]
  for (int i = 0; i < 10; ++i);

  // expected-error@+2 {{incompatible loop unroll instructions: '#pragma unroll(4)' and '[[clang::loop_unroll(2)]]'}}
#pragma unroll 4
  [[clang::loop_unroll(2)]]
  for (int i = 0; i < 10; ++i);

  // no error expected
  [[clang::loop_unroll(4)]]
  [[intel::initiation_interval(2)]] for (int i = 0; i < 10; ++i);

  // expected-error@+2 {{'loop_unroll' attribute requires an integer constant}}
  int b = 4;
  [[clang::loop_unroll(b)]]
  for (int i = 0; i < 10; ++i);

  // no error expected
  constexpr int c = 4;
  [[clang::loop_unroll(c)]]
  for (int i = 0; i < 10; ++i);

  // expected-note@+1 {{in instantiation of function template specialization}}
  bar<-4>();

  // no error expected
  bar<c>();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
  });
  return 0;
}
