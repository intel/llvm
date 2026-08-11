// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

// Baseline semantics for declcall (P2825): overload resolution, operand
// requirements, and dependent-operand instantiation.

int f(int);
int f(char);
using FPInt = int (*)(int);
using FPChar = int (*)(char);

// declcall(f(0)) resolves the overload as if calling f(0) and yields a
// pointer to the selected function.
constexpr auto p = declcall(f(0));
static_assert(__is_same(decltype(p), int (*const)(int)));
static_assert(p == static_cast<FPInt>(f));

// A char argument selects the other overload.
constexpr auto pc = declcall(f('a'));
static_assert(__is_same(decltype(pc), int (*const)(char)));

// The operand must be a call expression.
int x;
auto bad = declcall(x); // expected-error {{declcall doesn't contain a call}}

// A call through a runtime function pointer is not constant-evaluable. This
// must produce exactly one diagnostic: the failed declcall must not leave a
// live expression behind that cascades into further errors.
int (*fp)(int) = &f;
constexpr auto rt = declcall(fp(0)); // expected-error {{declcall must not depend on runtime known value}}

// A call through a pointer to member has no compile-time-known callee. This
// must be diagnosed, not crash the compiler.
struct S { int m(int); };
S s;
int (S::*pmf)(int);
auto pm = declcall((s.*pmf)(0)); // expected-error {{declcall doesn't support a call through a pointer to member yet}}

// Dependent operands are resolved at instantiation time.
template <class T> auto call(T v) { return declcall(f(v)); }
void use() {
  static_assert(__is_same(decltype(call(0)), FPInt));
  static_assert(__is_same(decltype(call('a')), FPChar));
}
