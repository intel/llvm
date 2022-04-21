// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// Test to ensure that template instantiation of an invalid attribute argument
// does not result in a null pointer crash when rebuilding the attributed
// statement.
template <int A> void bar() {
  // expected-error@+1 {{'loop_unroll' attribute requires a positive integral compile time constant expression}}
  [[clang::loop_unroll(A)]] for (int i = 0; i < 10; ++i);
}

void foo() {
  bar<-1>(); // expected-note {{in instantiation of function template specialization 'bar<-1>' requested here}}
}
