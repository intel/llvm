// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

// Diagnostic tests for __attribute__((io_pipe_id(..))) attribute.

// no error expected
using type1 = __attribute__((pipe("read_only"))) const int;

// no error expected
using type2 = __attribute__((pipe("write_only"))) const int;

// expected-error@+1 {{'42' mode for pipe attribute is not supported; allowed modes are 'read_only' and 'write_only'}}
using type3 = __attribute__((pipe("42"))) const int;

// expected-error@+1{{'pipe' attribute requires a string}}
using type4 = __attribute__((pipe(0))) const int;

// expected-error@+1{{'pipe' attribute takes one argument}}
using type5 = __attribute__((pipe)) const int;

struct pipe_storage {};

// no error expected
const pipe_storage Storage1 __attribute__((io_pipe_id(1)));

// expected-error@+1{{'io_pipe_id' attribute requires a non-negative integral compile time constant expression}}
const pipe_storage Storage2 __attribute__((io_pipe_id(-11)));

// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}
const pipe_storage Storage3 __attribute__((io_pipe_id("abc")));

// expected-error@+1{{'io_pipe_id' attribute only applies to SYCL pipe storage declaration}}
int Storage4 __attribute__((io_pipe_id(5)));

// expected-error@+2{{'io_pipe_id' attribute requires a non-negative integral compile time constant expression}}
template <int N>
pipe_storage Storage5 __attribute__((io_pipe_id(N)));

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
const pipe_storage Storage6 __attribute__((io_pipe_id(Ty{})));

struct S {};

// Test that checks template instantiations for different arg values.
template <int A>
pipe_storage Storage7 __attribute__((io_pipe_id(1))) // expected-note {{previous attribute is here}}
__attribute__((io_pipe_id(A)));                      // expected-warning{{attribute 'io_pipe_id' is already applied with different arguments}}

void foo(pipe_storage PS) {}

int main() {
  // no error expected
  foo(Storage5<2>);
  // expected-note@+1{{in instantiation of variable template specialization 'Storage5' requested here}}
  foo(Storage5<-1>);
  // expected-note@+1{{in instantiation of variable template specialization 'Storage6' requested here}}
  foo(Storage6<S>);
  // expected-note@+1{{in instantiation of variable template specialization 'Storage7' requested here}}
  foo(Storage7<4>);
  return 0;
}

// Test for Intel 'io_pipe_id' attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
const pipe_storage Storage8 __attribute__((io_pipe_id(1))) __attribute__((io_pipe_id(1)));

// Diagnostic is emitted because the arguments mismatch.
const pipe_storage Storage9 __attribute__((io_pipe_id(1))) // expected-note {{previous attribute is here}}
__attribute__((io_pipe_id(5)));                            // expected-warning{{attribute 'io_pipe_id' is already applied with different arguments}}

const pipe_storage Storage10 __attribute__((io_pipe_id(1)));        // expected-note {{conflicting attribute is here}}
extern const pipe_storage Storage10 __attribute__((io_pipe_id(3))); // expected-error{{attribute 'io_pipe_id' cannot appear more than once on a declaration}}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
const pipe_storage Storage11 __attribute__((io_pipe_id(baz() + 1)));

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
const pipe_storage Storage12 __attribute__((io_pipe_id(bar() + 2))); // OK
