// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

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

// expected-error@+1{{'io_pipe_id' attribute requires an integer constant}}
const pipe_storage Storage3 __attribute__((io_pipe_id("abc")));

// expected-error@+1{{'io_pipe_id' attribute only applies to SYCL pipe storage declaration}}
int Storage4 __attribute__((io_pipe_id(5)));

// expected-error@+2{{'io_pipe_id' attribute requires a non-negative integral compile time constant expression}}
template <int N>
pipe_storage Storage5 __attribute__((io_pipe_id(N)));

void foo(pipe_storage PS) {}

int main() {
  // no error expected
  foo(Storage5<2>);
  // expected-note@+1{{in instantiation of variable template specialization 'Storage5' requested here}}
  foo(Storage5<-1>);
}
