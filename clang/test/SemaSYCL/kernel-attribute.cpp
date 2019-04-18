// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

__kernel void foo();

// It's safe to use "kernel" for variable or function names:
void foo1(int kernel);
void kernel(int kernel);

// It's possible to apply __kernel to template function:
template <typename T>
__kernel void foo2(T a);

// expected-error@+1{{expected unqualified-id}}
kernel void foo3(); // expected-error{{unknown type name 'kernel'}}
