// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s

// This test checks that compiler doesn't crash if type of kernel argument is
// invalid.

#include "Inputs/sycl.hpp"

// Invalid field -> invalid decl
class A {
public:
  // expected-error@+1 {{unknown type name 'it'}}
  it KN;
};

// Invalid type of field -> invalid decl
class B {
  A Arr[100];
};

// Invalid base -> invalid decl
class E : public B {
};

// expected-note@+1 {{forward declaration of 'C'}}
class C;

// Invalid base -> invalid decl
// expected-error@+1 {{base class has incomplete type}}
class D : public B, C {
};

// Such thing is also invalid and caused crash
// expected-note@+1 {{definition of 'F' is not complete until the closing '}'}}
class F {
  // expected-error@+1 {{field has incomplete type 'F'}}
  F Self;
};

template <typename T>
class G {
  T Field;
};

class H {
  // expected-note@+1 {{previous declaration is here}}
  int A;
  // expected-error@+1 {{duplicate member 'A'}}
  int A;
};

int main() {
  A Obj{};
  D Obj1{};
  B Obj2{};
  E Obj3{};
  F Obj4{};
  G<A> Obj5{};
  H Obj6{};
  cl::sycl::kernel_single_task<class kernel>(
      [=]() {
        (void)Obj;
        (void)Obj1;
        (void)Obj2;
        (void)Obj3;
        (void)Obj4;
        (void)Obj5;
        (void)Obj6;
      });
  return 0;
}
