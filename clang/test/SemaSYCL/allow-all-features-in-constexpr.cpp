// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fcxx-exceptions -verify -fsyntax-only -std=c++23 -triple spir64 -aux-triple x86_64 %s

// The test checks that all SYCL device code limitations are lifted in
// manifestly constant-evaluated expressions under an option.
#include "sycl.hpp"

struct Base {
  virtual constexpr void f() {
  }
};

struct Derived : Base {
  constexpr void f() override {}
};

constexpr int recurse(int x) {
  if (x > 1)
    return 1 + recurse(x-1);
  return x;
}


constexpr int variadic(int p, ...) {
  return p;
}


int NonConstGlob = 1;
using myFuncDef = int(int);

// This function tries all constructs that are not allowed by SYCL 2020.
constexpr int convert(long double ld, myFuncDef fp = &recurse) {
  int *p = new int;
  delete p;
  recurse((int)ld);
  if (ld == 100)
    throw 1;
  Base *D = new Base;
  D->f();
  delete D;
  if (fp)
    fp(1);

  long double a;

  if (ld == 0) {
    // Thread local is not yet allowed in constexpr if unconditionally met, so
    // check under if.
    thread_local unsigned int R = 1;
    R++;
    NonConstGlob++;
  }
  variadic(1);

  return static_cast<int>(ld);
}

template <int A>
class B {
  int Array[A];
};

struct ConditionallyExplicitCtor {
  explicit(convert(103.L) == 103) ConditionallyExplicitCtor(int i) {}
};

void conditionally_noexcept() noexcept(static_cast<bool>(convert(5.L))) {}

consteval int callsConvert(long double ld) {
  return convert(ld);
}


void host(int n) {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    // This kernel calls code with SYCL 2020 restrictions from all cases of
    // manifestly constant-evaluated expressions listed in actual (in the
    // present moment when this test is being developed) version of C++
    // standard.
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by}}
    cgh.single_task([=] {
      constexpr int v2 [convert(2.L)] = {0};
      constexpr int v3 [convert(2.L)] = {convert(1.L)};
      B<convert(2.L)> BB;
      struct HasBitF {
        unsigned long : convert(3.L);
      };
      enum Foo { a, b, c = convert(10.L), d, e = 1, f, g = f + c };
      alignas(convert(64.L)) char Arrr[64];


      switch (1) {
        case convert(1.L): return;
      }

      static_assert(convert(1) == 1);
      if constexpr (convert(1.L)) {
      } else  {
        // No error in discarded statement too, it is unreachable anyway.
         new int;
      }

      ConditionallyExplicitCtor ccc(1);
      conditionally_noexcept();

      constexpr struct ConstExprObjInit {
        int a;
      } objfoo = {convert(1.L)};

      // with const int variables it is tricky, since the initializer is
      // manifestly constant-evaluated only if all parts of initializer are
      // contant expressions.
      static const int v4 = convert(1.L);
      const int v5 = convert(1.L);
      const Foo v6 = (Foo)convert(1.L);
      const int v7 = callsConvert(1.L);

      // report an error if not constexpr initializer.
      const int shooinfoo = (unsigned long long)(new int); // expected-error {{cannot allocate storage}}

    });
  });
}
