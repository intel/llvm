// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -fexperimental-new-constant-interpreter -fsyntax-only -verify %s
// expected-no-diagnostics

// Calling through a declcall member pointer at compile time must honor
// devirtualization consistently in both constant evaluators.

struct Base {
  constexpr virtual int foo() const { return 1; }
};
struct Derived : Base {
  constexpr int foo() const override { return 2; }
};

constexpr Derived d;

// A qualified virtual call devirtualizes: calling through the pointer bypasses
// virtual dispatch and invokes Base::foo (1).
constexpr auto devirt = declcall(((Base *)0)->Base::foo());
static_assert((d.*devirt)() == 1);

// An unqualified virtual call keeps virtual dispatch: Derived::foo (2).
constexpr auto virt = declcall(((Base *)0)->foo());
static_assert((d.*virt)() == 2);
